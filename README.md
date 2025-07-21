# MDR Stratify

AI-driven MDR (Multi-Drug Resistance) Prediction for Optimized Antibiotic Use in LMICs (Low and Middle Income Countries).

## Overview

MDR Stratify is a comprehensive web application that predicts the likelihood of Multi-Drug Resistance in pathogens based on patient demographics, clinical settings, and pathogen characteristics. The system uses machine learning to help healthcare providers make informed decisions about antibiotic treatment.

## Features

- **Patient Data Input**: Comprehensive form for entering patient information
- **Real-time Prediction**: Instant MDR risk assessment
- **Visual Results**: Clear, actionable prediction results with confidence levels
- **Responsive Design**: Works on desktop and mobile devices
- **API Integration**: Modular backend for easy integration with existing systems

## Technology Stack

- **Frontend**: Next.js 13, TypeScript, Tailwind CSS, shadcn/ui
- **Backend**: FastAPI (Python), scikit-learn, pandas
- **Infrastructure**: Docker, Docker Compose, Nginx
- **Build Tools**: Makefile for automation

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.9+
- Docker and Docker Compose

### Installation

1. Clone the repository and navigate to the project directory
2. Run the setup command:
   ```bash
   make setup
   ```

### Development

Start the development environment:
```bash
make docker-up
```

Or run individual services:
```bash
# Frontend only
make dev

# API only
cd services/api && python main.py
```

### Building

Build the application:
```bash
make build
```

Build Docker images:
```bash
make docker-build
```

## Input Parameters

### Patient Information
- **Age**: Patient age in years
- **Sex**: Male/Female
- **Country**: Country of origin
- **Year**: Year of data collection

### Clinical Context
- **Ward**: Hospital ward/department
- **Specimen Type**: Type of clinical specimen
- **Organism**: Identified pathogen
- **Antibiotics**: List of antibiotics tested/used

## Model Training

To train your own model:

1. Prepare your dataset with the required features
2. Train your model using scikit-learn or your preferred ML library
3. Save the trained models in the `models` folder
4. Update the preprocessing function in `services/api/main.py`

## Docker Services

The application consists of three main services:

- **mdr-app**: Next.js frontend application
- **mdr-api**: FastAPI backend service
- **nginx**: Reverse proxy and load balancer

## Available Commands

```bash
make help          # Show available commands
make install       # Install dependencies
make dev          # Start development server
make build        # Build the application
make docker-up    # Start Docker containers
make docker-down  # Stop Docker containers
make clean        # Clean build artifacts
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For support and questions, please refer to the project documentation or create an issue in the repository.