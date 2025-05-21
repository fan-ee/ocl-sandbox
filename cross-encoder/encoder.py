from typing import List, Tuple
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

import torch
from tqdm import tqdm

# LOINC lab test codes and their descriptions
candidate_pool = [
    ("718-7", "Hemoglobin [Mass/volume] in Blood"),
    ("4544-3", "Hematocrit [Volume Fraction] of Blood by Automated count"),
    ("751-8", "Neutrophils [#/volume] in Blood by Automated count"),
    ("731-0", "Lymphocytes [#/volume] in Blood by Automated count"),
    ("785-6", "MCH [Entitic mass] by Automated count"),
    ("786-4", "MCHC [Mass/volume] by Automated count"),
    ("787-2", "MCV [Entitic volume] by Automated count"),
    ("788-0", "Erythrocyte distribution width [Ratio] by Automated count"),
    ("777-3", "Platelets [#/volume] in Blood by Automated count"),
    ("2345-7", "Glucose [Mass/volume] in Serum or Plasma"),
    ("2160-0", "Creatinine [Mass/volume] in Serum or Plasma"),
    ("3094-0", "BUN [Mass/volume] in Serum or Plasma"),
    ("2093-3", "Cholesterol [Mass/volume] in Serum or Plasma"),
    ("2571-8", "Triglyceride [Mass/volume] in Serum or Plasma"),
    ("1920-8", "AST [Enzymatic activity/volume] in Serum or Plasma"),
    ("1742-6",
     "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma"),
    ("6768-6",
     "Alkaline phosphatase [Enzymatic activity/volume] in Serum or Plasma"),
    ("1975-2", "Total Bilirubin [Mass/volume] in Serum or Plasma"),
    ("2085-9", "HDL Cholesterol [Mass/volume] in Serum or Plasma"),
    ("2089-1", "LDL Cholesterol [Mass/volume] in Serum or Plasma")
]


class MedicalTermRanker:
    def __init__(self, candidate_pool, method: str):
        """Initialize the cross-encoder model for medical term ranking."""
        self.candidates = candidate_pool
        if method == 'cross':
            model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_name)
            self._score_method = self.cross_encoder_score

        if method == 'bi':
            self._model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self._score_method = self.bi_encoder_score

    def bi_encoder_score(self, query: str, candidate: str) -> float:
        """Get similarity score between query and candidate using bi-encoder."""
        query_vec = self._model.encode(str(query))
        candidate_vec = self._model.encode(str(candidate))
        similarities = self._model.similarity(query_vec, candidate_vec)
        score = float(similarities.numpy()[0][0])
        return score

    def cross_encoder_score(self, query: str, candidate: str) -> float:
        """Get similarity score between query and candidate using cross-encoder."""
        
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._model.to(device)
        # Add deterministic setting for better reproducibility
        torch.backends.cudnn.deterministic = True

        inputs = self._tokenizer(
            [query],
            [candidate],
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            scores = outputs.logits.squeeze()
            return float(scores.cpu().numpy())

    def rank_terms(self, description: str) -> List[Tuple[str, str, float]]:
        """
        Rank LOINC terms based on their relevance to the input description.

        Args:
            description (str): Free-form text description of the medical concept

        Returns:
            List[Tuple[str, str, float]]: Ranked list of (LOINC code, description, score)
        """
        # Get similarity scores for each candidate
        results = [
            (code, desc, self._score_method(description, desc))
            for code, desc in tqdm(self.candidates, desc='Ranking terms')
        ]

        # Sort by score in descending order
        ranked_results = sorted(results, key=lambda x: x[2], reverse=True)
        return ranked_results

    def print_ranked_results(self, ranked_results: List[Tuple[str, str, float]]):
        """Print ranked results in a formatted way."""
        df = pd.DataFrame(ranked_results, columns=[
                          'LOINC Code', 'Description', 'Score'])
        df['Score'] = df['Score'].round(4)
        print("\nRanked Results:")
        print(df.to_string(index=False))
