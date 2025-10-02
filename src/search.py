"""
Módulo de Busca e Sumarização
Implementa busca por similaridade e sumarização de contexto para o sistema RAG
"""

import os
import logging
from typing import List

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Carrega variáveis de ambiente
load_dotenv()


# Configuração de logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuração de logging
logger = logging.getLogger(__name__)


class DocumentSearchTool:
    """Ferramenta para busca por similaridade e sumarização de contexto"""
    
    def __init__(self, collection_name: str = "pdf_documents"):
        """Inicializa a ferramenta de busca"""
        try:
            self.collection_name = collection_name
            
            # Inicializa embeddings
            self.embeddings = OpenAIEmbeddings(
                model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
                api_key=os.getenv('OPENAI_API_KEY')
            )
            
            # Inicializa LLM para sumarização
            self.llm = ChatOpenAI(
                model=os.getenv('CHAT_MODEL', 'gpt-5-nano'),
                api_key=os.getenv('OPENAI_API_KEY'),
                temperature=0.1
            )
            
            # Inicializa vector store
            self.vectorstore = PGVector(
                collection_name=collection_name,
                connection=os.getenv('DATABASE_URL'),
                embeddings=self.embeddings,
                use_jsonb=True
            )
            
            # Cria chain de sumarização
            self._setup_summarization_chain()
            
            logger.info(f"DocumentSearchTool inicializada para coleção: {collection_name}")
            
        except Exception as e:
            logger.error(f"Falha ao inicializar DocumentSearchTool: {str(e)}")
            raise
    
    def _setup_summarization_chain(self):
        """Configura a chain de sumarização usando pipes"""
        summarization_prompt = PromptTemplate(
            input_variables=["documents", "query"],
            template="""
            You are a helpful AI assistant that summarizes document chunks to provide relevant context for answering questions.
            
            Query: {query}
            
            Document chunks:
            {documents}
            
            Please provide a concise summary of the most relevant information from these document chunks that would help answer the query. 
            Focus on key facts, concepts, and details that directly relate to the question.
            If the documents don't contain relevant information, say so clearly.
            
            Summary:
            """
        )
        
        # Cria chain usando pipes (abordagem moderna do LangChain)
        self.summarization_chain = (
            summarization_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        logger.info("Configuração da chain de sumarização concluída")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Executa busca por similaridade no vector store"""
        try:
            logger.info(f"Executando busca por similaridade para: '{query}' (k={k})")
            
            # Executa busca por similaridade
            results = self.vectorstore.similarity_search(
                query=query,
                k=k
            )
                        
            logger.info(f"Encontrados {len(results)} documentos similares")
            return results
            
        except Exception as e:
            logger.error(f"Busca por similaridade falhou: {str(e)}")
            raise
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Executa busca por similaridade com pontuações de relevância"""
        try:
            logger.info(f"Executando busca por similaridade com pontuações para: '{query}' (k={k})")
            
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k
            )
            
            logger.info(f"Encontrados {len(results)} documentos com pontuações")
            return results
            
        except Exception as e:
            logger.error(f"Busca por similaridade com pontuações falhou: {str(e)}")
            raise
    
    def summarize_context(self, documents: List[Document], query: str) -> str:
        """Sumariza documentos recuperados para contexto"""
        try:
            if not documents:
                return "Nenhum documento relevante encontrado."
            
            logger.info(f"Sumarizando {len(documents)} documentos")
            
            # Formata documentos para sumarização
            doc_texts = []
            for i, doc in enumerate(documents, 1):
                metadata_info = ""
                if doc.metadata:
                    source = doc.metadata.get('source_file', 'Unknown')
                    chunk_id = doc.metadata.get('chunk_id', 'Unknown')
                    metadata_info = f"[Source: {source}, Chunk: {chunk_id}]"
                
                doc_texts.append(f"Document {i} {metadata_info}:\n{doc.page_content}\n")
            
            documents_text = "\n".join(doc_texts)
            
            # Gera sumário usando a chain
            summary = self.summarization_chain.invoke({
                "documents": documents_text,
                "query": query
            })
            
            logger.info("Sumarização de contexto concluída")
            return summary
            
        except Exception as e:
            logger.error(f"Sumarização de contexto falhou: {str(e)}")
            return f"Erro ao sumarizar contexto: {str(e)}"
    
    def search_and_summarize(self, query: str, k: int = 5) -> str:
        """Pipeline completo de busca e sumarização"""
        try:
            # Executa busca por similaridade
            documents = self.similarity_search(query, k)

            # Sumariza o contexto recuperado
            summary = self.summarize_context(documents, query)
            
            return summary
            
        except Exception as e:
            logger.error(f"Pipeline de busca e sumarização falhou: {str(e)}")
            return f"Erro na busca e sumarização: {str(e)}"


def create_search_tool(collection_name: str = "pdf_documents") -> tool:
    """Cria uma Ferramenta LangChain para busca de documentos"""
    try:
        search_tool = DocumentSearchTool(collection_name)
        
        # Cria Ferramenta LangChain (decorator)
        @tool("document_search",description="""
            Use this tool to search through the ingested PDF documents for information relevant to answering questions.
            The tool will perform similarity search and return a summarized context that's most relevant to your query.
           Input should be a clear question or search query about the document content.
            """,return_direct=True)
        def document_search(query: str) -> str:
            """Busca documentos e retorna contexto sumarizado"""
            try:
                logger.info(f"Ferramenta executando busca para: '{query}'")
                summary = search_tool.search_and_summarize(query, k=10)

                quest_prompt = PromptTemplate(
                    input_variables=["summary", "query"],
                    template="""                        
                        CONTEXTO:
                        
                        {summary}

                        REGRAS:
                        - Responda somente com base no CONTEXTO.
                        - Se a informação não estiver explicitamente no CONTEXTO, responda:
                        "Não tenho informações necessárias para responder sua pergunta."
                        - Nunca invente ou use conhecimento externo.
                        - Nunca produza opiniões ou interpretações além do que está escrito.

                        EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
                        Pergunta: "Qual é a capital da França?"
                        Resposta: "Não tenho informações necessárias para responder sua pergunta."

                        Pergunta: "Quantos clientes temos em 2024?"
                        Resposta: "Não tenho informações necessárias para responder sua pergunta."

                        Pergunta: "Você acha isso bom ou ruim?"
                        Resposta: "Não tenho informações necessárias para responder sua pergunta."

                        PERGUNTA DO USUÁRIO:
                        {query}

                        RESPONDA A "PERGUNTA DO USUÁRIO"                    
                    """
                )

                quest_chain = quest_prompt | search_tool.llm | StrOutputParser()

                result = quest_chain.invoke({"summary":summary,"query": query})

                logger.info("Busca da ferramenta concluída com sucesso")
                return result
            except Exception as e:
                error_msg = f"Erro na ferramenta de busca: {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        
       # Cria Ferramenta LangChain (sem decorator) retorno -> Tool ao invés tool
        #tool = Tool(
         #   name="document_search",
          #  description="""
          #  Use this tool to search through the ingested PDF documents for information relevant to answering questions.
           # The tool will perform similarity search and return a summarized context that's most relevant to your query.
          # Input should be a clear question or search query about the document content.
          #  """,
            #func=search_documents
       # )
        
        
        logger.info("Ferramenta de busca criada com sucesso")
        return document_search
        
    except Exception as e:
        logger.error(f"Falha ao criar ferramenta de busca: {str(e)}")
        raise

def search_pdf():
    """Função de teste para executar search no PDF"""
    
    # Pega a Collection 
    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME")
    my_tool = create_search_tool(collection_name)

    result = my_tool.invoke({"query": "Qual o faturamento da Empresa SuperTechIABrazil?"})

    logger.info("Resultado : " +result)


if __name__ == "__main__":
    search_pdf()


""""
Exemplo de Saida :

2025-09-19 14:47:29,482 - INFO - Configuração da chain de sumarização concluída
2025-09-19 14:47:29,482 - INFO - DocumentSearchTool inicializada para coleção: fullcycle_langchain
2025-09-19 14:47:29,484 - INFO - Ferramenta de busca criada com sucesso
2025-09-19 14:47:29,487 - INFO - Ferramenta executando busca para: 'Qual o faturamento da Empresa SuperTechIABrazil?'
2025-09-19 14:47:29,487 - INFO - Executando busca por similaridade para: 'Qual o faturamento da Empresa SuperTechIABrazil?' (k=10)
2025-09-19 14:47:30,378 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 503 Service Unavailable"
2025-09-19 14:47:30,379 - INFO - Retrying request to /embeddings in 0.492665 seconds
2025-09-19 14:47:31,830 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
2025-09-19 14:47:31,865 - INFO - Encontrados 10 documentos similares
2025-09-19 14:47:31,865 - INFO - Sumarizando 10 documentos
2025-09-19 14:47:33,546 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-09-19 14:47:33,564 - INFO - Sumarização de contexto concluída
2025-09-19 14:47:33,564 - INFO - Busca da ferramenta concluída com sucesso
2025-09-19 14:47:33,564 - INFO - Resultado : A Empresa SuperTechIABrazil teve um faturamento de R$ 10.000.000,00 em 2025, conforme indicado nos documentos. Não há outras informações relevantes 
sobre a empresa nos documentos fornecidos
"""