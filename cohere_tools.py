import os
from dotenv import load_dotenv

import cohere
from langchain.embeddings import CohereEmbeddings

load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")


class CohereTools:
    def __init__(self, api_key):
        self.api = api_key
        self._client = cohere.Client(api_key=self.api)
        self._embeddings = CohereEmbeddings(cohere_api_key=self.api)

    def email_classify(self, email: str, model: str) -> str:
        """
        Classifies an email using the specified model.
        Args:
        - email (str): The email text to be classified.
        - model (str): The ID of the model to use for classification.
        Returns:
        - str: The predicted class label for the email.
        """
        response = self._client.classify(
            model=model,
            inputs=[email],
        ).classifications[0]

        return response.prediction

    def vector_present(self, text: str):
        """
        Computes a vector representation of the input text.
        Args:
        - text (str): The input text to be transformed into a vector.
        Returns:
        - np.ndarray: A numpy array representing the vectorized text.
        """
        embeddings = self._embeddings.embed_query(text)
        return embeddings


if __name__ == "__main__":
    model_id = '689b50b6-64cc-4dc0-822c-306676d82c19-ft'
    email1 = "Welcome to your latest Career Supplement, the bi-weekly email that takes just 5 minutes to read, but " \
           "contains stuff that will fast-track your career by years. 🔔 Today's email is sponsored by Notion. " \
           "It's the productivity workspace that keeps our team organized and efficient. We use it for" \
           " everything — managing tasks, creating docs, you name it. Try it out for free here."
    email2 = "I can't believe but our team works very fast"

    cohere_init = CohereTools(cohere_api_key)
    print(f'First email: {cohere_init.email_classify(email1, model_id)}')
    print(f'First email: {cohere_init.email_classify(email2, model_id)}')


    print(cohere_init.vector_present(email2))
