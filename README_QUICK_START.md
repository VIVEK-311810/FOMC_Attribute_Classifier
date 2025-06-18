# FOMC Statement Classifier 🏛️

A web application for classifying Federal Open Market Committee (FOMC) statements using fine-tuned FinBERT models across six key financial attributes.

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Historical data Excel file in the `data/` directory

### Directory Structure
```
fomc_classifier/
├── data/
│   └── Sentiment&Attributes_Classification.xlsx
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── start.sh
```

### Local Deployment

1. **Clone and Setup**
   ```bash
   git clone <your-repo>
   cd fomc_classifier
   ```

2. **Add Your Data**
   - Copy your Excel file to `data/Sentiment&Attributes_Classification.xlsx`

3. **Run with Docker Compose**
   ```bash
   docker-compose up
   ```

4. **Access the Application**
   - Streamlit App: http://localhost:8501

### Cloud Deployment

#### Hugging Face Spaces (Recommended)
1. Create a new Space on Hugging Face
2. Choose "Streamlit" as SDK
3. Push your code to the Space repository
4. The application will build and deploy automatically

#### Other Platforms
- **Render.com**: Connect GitHub repo, select Docker environment
- **Google Cloud Run**: Build image and deploy to Cloud Run
- **AWS Fargate**: Use ECS with Fargate launch type

## Features

- **Interactive Classification**: Input FOMC statement text and get predictions across 6 attributes
- **Historical Data Browser**: Explore historical FOMC statements by year and month
- **Confidence Scoring**: Color-coded confidence indicators for predictions
- **Responsive Design**: Works on desktop and mobile devices

## Classification Attributes

| Attribute | Labels |
|-----------|--------|
| **Sentiment** | Positive, Neutral, Negative |
| **Economic Growth** | UP, Down, Flat |
| **Employment Growth** | UP, Down, Flat |
| **Inflation** | UP, Down, Flat |
| **Medium Term Rate** | Hawk, Dove |
| **Policy Rate** | Raise, Flat, Lower |

## Architecture

- **Application**: Single Streamlit application with integrated models and data handling
- **Models**: Six fine-tuned FinBERT models for attribute classification (loaded from Hugging Face)
- **Data**: Historical FOMC statements with labeled attributes
- **Deployment**: Docker containerization for consistent deployment

## Development

### Local Development (Python)

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Streamlit App**
   ```bash
   streamlit run app.py
   ```

### Environment Variables

- `EXCEL_PATH`: Path to Excel file (default: `./data/Sentiment&Attributes_Classification.xlsx`)

## Troubleshooting

### Common Issues

1. **Models not loading**: Ensure internet connectivity for Hugging Face model downloads.
2. **Historical data not available**: Verify Excel file path and format.
3. **Memory issues**: Increase Docker memory allocation (8GB+ recommended).

## Performance

- **Memory**: 8GB+ recommended for all models
- **CPU**: 4+ cores recommended
- **GPU**: Optional, can improve inference speed
- **Startup**: 2-3 minutes for model loading

## Security

- Input validation and sanitization
- No persistent storage of user inputs
- HTTPS support in production deployments

## Documentation

See `DEPLOYMENT_GUIDE.md` for comprehensive deployment instructions, including:
- Detailed setup procedures
- Cloud platform specific guidance
- Production considerations
- Performance optimization
- Security best practices

## License

MIT License - see LICENSE file for details

## Citation

If you use this application in your research, please cite:

```bibtex
@software{fomc_classifier_2024,
  title={FOMC Statement Classifier},
  author={Vivek Maddula},
  year={2024},
  url={https://github.com/your-repo/fomc-classifier}
}
```

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the comprehensive deployment guide
3. Open an issue in the repository
4. Contact the development team

---

**Built with ❤️ for financial NLP research**



