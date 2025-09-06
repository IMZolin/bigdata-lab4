import re
import string
import numpy as np
from src.logger import Logger

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)  
    text = re.sub(r"@\w+", "", text)  
    text = re.sub(r"#\w+", "", text)  
    text = text.translate(str.maketrans("", "", string.punctuation))  
    return text.lower().strip()

def prepare_text(text_series, vectorizer):
    return vectorizer.fit_transform(text_series).toarray()

def load_config(path="config.ini"):
    try:
        logger.info(f"Loading config from: {path}")
        config = configparser.ConfigParser()
        config.read(path)
        if not config.sections():
            logger.warning(f"Config file {path} is empty or not found")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise