# -------------------------------------------------
# Dependencies ----------------------------------
# -------------------------------------------------

## Libraries

# Generic
import os
import sys
import glob
import base64
import csv
import random
import time
import math
import uuid
import json
import re
import string
import json
import pickle

# Database
import pyodbc

# Analysis
import pandas as pd
import numpy as np

# ML & Co
import scipy
import sklearn

## pre-processing
from   sklearn.feature_extraction.text import TfidfVectorizer

## model architecture
from sklearn.multiclass import OneVsRestClassifier

## models
from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from keras.models import Sequential
from keras import layers

## tuning, etc.
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

## metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss


# NLP
import nltk
import spacy
import pycorenlp
from   pycorenlp.corenlp import StanfordCoreNLP
from   langdetect import detect
from   vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
