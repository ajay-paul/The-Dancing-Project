# The-Dancing-Project

## LSTM VAE
```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'darkMode': true }}}%%
graph TD
    %% Block 1: Encoder
    A[MFCC Input shape: time_steps, n_mfcc] --> B[LSTM Layer 128 units]
    B --> C[LSTM Layer 128 units]
    C --> D{Latent Space}
    
    %% Latent Space: z_mean, z_log_var, sampling
    D -->|z_mean| E[z_mean latent_dim]
    D -->|z_log_var| F[z_log_var latent_dim]
    E --> G[sampling z_mean, z_log_var]
    F --> G
    
    %% Block 2: Decoder
    G --> H[Dense Layer 128 units]
    H --> I[Repeat Vector time_steps]
    I --> J[LSTM Layer 128 units]
    J --> K[TimeDistributed Dense Layer]
    K --> L[Pose Output shape: 33, 2]

    %% Input and Output shape description
    classDef input_output fill:#ffdd57,stroke:#333,stroke-width:4px;

    %% Flow diagram
    subgraph Flow Diagram
    B -->|Encodes| G -->|Decodes| L
    end

    %% Loss function calculation
    subgraph Loss Function Calculation
    Z[Reconstruction Loss MSE] --> L
    Y[KL Divergence Loss] --> D
    Total_Loss --> Z & Y
    end

    %% Training and Inference process
    subgraph Training and Inference
    MFCC_Input --> Encoder --> Latent_Space --> Decoder --> Pose_Output
    end

    %% Custom Colors for Nodes and Links
    classDef latent_space fill:#7ed6df,stroke:#fff,stroke-width:2px;
    classDef lstm_layer fill:#4834d4,stroke:#fff,stroke-width:2px;
    classDef dense_layer fill:#f0932b,stroke:#fff,stroke-width:2px;
    classDef time_distributed fill:#6ab04c,stroke:#fff,stroke-width:2px;
    classDef sampling fill:#2c3e50,stroke:#fff,stroke-width:2px;  %% Darker background for sampling block

    %% Apply Styles
    H:::dense_layer
    G:::sampling
    K:::time_distributed

    %% Customize Flow Line Width and Colors
    linkStyle default stroke:#7a7aff,stroke-width:4px
    linkStyle 0 stroke-width:4px
    linkStyle 1 stroke-width:4px
```

## GAN Model

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'darkMode': true }}}%%
graph TD
    %% Generator Section
    A[Pose Data] --> B[Generator]
    B --> B1["Dense Layer (256 units) + ReLU"]
    B1 --> B2["Reshape (16x16x256)"]
    B2 --> B3["Conv2DTranspose (128 filters, 3x3) + ReLU"]
    B3 --> B4["Conv2DTranspose (64 filters, 3x3) + ReLU"]
    B4 --> B5["Conv2DTranspose (3 filters, 3x3) + Tanh"]
    B5 --> C[Generated Image]

    %% Discriminator Section
    C --> D[Discriminator]
    D --> D1["Conv2D (64 filters, 3x3) + LeakyReLU"]
    D1 --> D2["Conv2D (128 filters, 3x3) + LeakyReLU"]
    D2 --> D3["Flatten"]
    D3 --> D4["Dense Layer (1 unit, Sigmoid)"]
    D4 --> E{Discriminator Decision}
    E -->|Real| F[Real Image]
    E -->|Fake| G[Generator Loss]
    
    %% Training Process
    H[Start Training] --> I[Process Each Batch]
    I --> J[Generate Fake Images]
    J --> K[Send Real & Fake Images to Discriminator]
    K --> L[Discriminator Output]
    L --> M[Update Discriminator]
    M --> N[Calculate Loss]
    N --> O[Update Generator Weights]
    O --> P[Track Progress]
    P --> Q[End Epoch]

    %% Styling Definitions
    classDef default fill:#2a2a2a,stroke:#7a7a7a,color:#e0e0e0;
    classDef model fill:#4a4a8c,stroke:#7a7aff,color:#ffffff,stroke-width:2px;
    classDef output fill:#ffdd57,stroke:#333,stroke-width:2px;
    classDef loss fill:#f0932b,stroke:#fff,stroke-width:2px;
    classDef action fill:#374151,stroke:#6b7280,color:#ffffff;

    %% Apply Classes
    class A,B,B1,B2,B3,B4,B5,C,D,D1,D2,D3,D4 model;
    class E,F,G loss;
    class H,I,J,K,L,M,N,O,P,Q action;
```
