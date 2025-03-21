import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class PDFSearch:
    def __init__(self):
        self.file_path = "multimodal/data/article_2502.15214v1.pdf"
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.text = self.load_pdf()
        self.chunks = self.chunk_text()
        self.embeddings = self.calc_embeddings()
        self.index = self.faiss_index()
        

    def load_pdf(self) -> str:
        """
        This function loads the pdf file and extracts the text from it.
        """
        try:
            with open(self.file_path, "rb") as file:
                pdf = PyPDF2.PdfFileReader(file)
                text = ""
                for page_num in range(pdf.getNumPages()): #reads the pages one by one
                    text += pdf.getPage(page_num).extract_text()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not open PDF file at {self.file_path}: {e}")
        return text
    
    def chunk_text(self, chunk_size=1500) -> List[str]: #chose to have 1,5k char chunck sizes
        """
        Simple function to split the text into chunks of 1500 characters.
        """
        chunks = []
        for i in range(0, len(self.text), chunk_size):
            chunks.append(self.text[i:i + chunk_size])
        return chunks

    def calc_embeddings(self) -> np.ndarray:
        """
        Simple function to calculate the embeddings for the chunks of text.
        """
        embeddings = self.model.encode(self.chunks)
        return embeddings

    def faiss_index(self) -> faiss.IndexFlatL2:
        """
        Simple function to create a faiss index for the embeddings.
        """
        if not hasattr(self.embeddings, "shape"):
            raise ValueError("Embeddings do not have a shape attribute.")
        index = faiss.IndexFlatL2(self.embeddings.shape[1])
        index.add(self.embeddings)
        return index

    def search_query(self, query: str) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """
        Method that embeds a query and searches for the closest text extracts in the index.
        """
        query_embedding = self.model.encode([query]) #encode the query to be able to compare it with the text extracts
        distance, indices = self.index.search(query_embedding, k=2) #chose to keep the top 2 closest text extracts
        return [self.embeddings[i] for i in indices[0]] , indices[0], distance[0]

if __name__ == '__main__':
    #example usage of the PDFSearch class
    module = PDFSearch()
    query = "What is the difference between LLM and VLM?"
    _ , indices, distance = module.search_query(query)
    reversed_sentences = [module.chunks[i] for i in indices]
    print("Calculated closest text extracts:", reversed_sentences)
    print("With distances:", distance)