import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ASPP_Multimodal(nn.Module):
    def __init__(self, in_channels, tabular_dim, out_channels=256):
        super().__init__()
        
        # 1. Ramas de Convolución Dilatada (Atrous)
        # Esto expande el campo receptivo sin encoger la imagen
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv3x3_d6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False)
        self.conv3x3_d12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False)
        self.conv3x3_d18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False)
        
        # 2. Rama de Contexto Global (Global Average Pooling)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        
        # 3. Rama Tabular (Tu innovación Multimodal)
        # Convertimos los datos del ciclón (viento, presión) en un vector semántico
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels),
            nn.ReLU()
        )
        
        # 4. Compresión Final
        # Sumamos 5 ramas visuales + 1 rama tabular = 6 canales a concatenar
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 6, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x, tabular_data):
        # x shape: [Batch, Channels, Height, Width]
        
        # Procesamos la imagen en las 5 ramas de visión
        feat1 = self.conv1x1(x)
        feat2 = self.conv3x3_d6(x)
        feat3 = self.conv3x3_d12(x)
        feat4 = self.conv3x3_d18(x)
        
        feat5 = self.global_avg_pool(x)
        feat5 = F.interpolate(feat5, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Procesamos los datos atmosféricos (1D -> 2D)
        # tabular_data shape: [Batch, Tabular_Dim]
        tab_feat = self.tabular_mlp(tabular_data) # Shape: [Batch, 256]
        
        # Expandimos el vector tabular para que coincida con el tamaño espacial de la imagen
        # Shape: [Batch, 256, Height, Width]
        tab_feat_spatial = tab_feat.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2], x.shape[3])
        
        # Concatenamos absolutamente toda la información
        concatenated = torch.cat([feat1, feat2, feat3, feat4, feat5, tab_feat_spatial], dim=1)
        
        # Comprimimos de vuelta a 256 canales
        return self.project(concatenated)

class MultimodalDeepLabV3Plus(nn.Module):
    def __init__(self, spatial_channels=2, tabular_dim=3, num_classes=1):
        super().__init__()
        
        # 1. Backbone (ResNet34)
        # Modificamos la primera capa para aceptar tu topografía+lluvia (2 canales en vez de RGB)
        resnet = models.resnet34(weights=None)
        self.stem = nn.Conv2d(spatial_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1 # Low-level features (Bordes finos)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4 # High-level features (Contexto global)
        
        # 2. El Cerebro Multimodal
        self.aspp = ASPP_Multimodal(in_channels=512, tabular_dim=tabular_dim, out_channels=256)
        
        # 3. Decodificador (Recuperando los bordes exactos)
        # Reducimos los features de bajo nivel para que no dominen la suma
        self.low_level_project = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        
        # Fusionamos el ASPP con los bordes finos y sacamos la predicción final
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, spatial_x, tabular_x):
        # --- ENCODER ---
        x = self.stem(spatial_x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        low_level_features = self.layer1(x) # Extraemos los bordes limpios aquí
        
        x = self.layer2(low_level_features)
        x = self.layer3(x)
        x = self.layer4(x) # Aquí el mapa está altamente comprimido
        
        # --- ASPP MULTIMODAL ---
        # El ASPP inyecta dilataciones y la presión/viento para entender TODO el mapa
        aspp_features = self.aspp(x, tabular_x)
        
        # --- DECODER ---
        # Proyectamos los bordes finos
        low_level_features = self.low_level_project(low_level_features)
        
        # Ampliamos la salida del ASPP para que coincida con la resolución de los bordes finos
        aspp_features = F.interpolate(aspp_features, size=low_level_features.shape[2:], mode='bilinear', align_corners=False)
        
        # Los concatenamos: Contexto Multimodal Inteligente + Bordes Topográficos Perfectos
        concat_features = torch.cat([aspp_features, low_level_features], dim=1)
        
        out = self.decoder(concat_features)
        
        # Finalmente, lo escalamos de regreso al tamaño original de 512x512
        out = F.interpolate(out, size=spatial_x.shape[2:], mode='bilinear', align_corners=False)
        
        return out