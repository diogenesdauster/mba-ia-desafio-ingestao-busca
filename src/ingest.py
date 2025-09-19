"""
Script de Ingestão de PDFs
Carrega documentos PDF, divide em chunks, cria embeddings e armazena no PGVector
"""
import os
import sys
import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document

# Carrega variáveis de ambiente
load_dotenv()

# Configuração de logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console = Console()


class PDFIngestor:
    """Gerencia a ingestão de documentos PDF no banco de dados PGVector"""
    
    def __init__(self):
        """Inicializa o PDFIngestor com os componentes necessários"""
        try:
            # Inicializa embeddings
            self.embeddings = OpenAIEmbeddings(
                model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
                api_key=os.getenv('OPENAI_API_KEY')
            )
            
            # Inicializa divisor de texto
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=int(os.getenv('CHUNK_SIZE', 1000)),
                chunk_overlap=int(os.getenv('CHUNK_OVERLAP', 150)),
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Conexão com banco de dados
            self.connection_string = os.getenv('DATABASE_URL')
            if not self.connection_string:
                raise ValueError("A variável de ambiente DATABASE_URL é obrigatória")
                
            logger.info("PDFIngestor inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Falha ao inicializar PDFIngestor: {str(e)}")
            raise
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Carrega e analisa documento PDF"""
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_path}")
            
            logger.info(f"Carregando PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            logger.info(f"Carregadas {len(documents)} páginas do PDF")
            return documents
            
        except Exception as e:
            logger.error(f"Falha ao carregar PDF {pdf_path}: {str(e)}")
            raise
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Divide documentos em chunks"""
        try:
            logger.info("Dividindo documentos em chunks...")
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Criados {len(chunks)} chunks")
            enriched = [
                Document(
                page_content=d.page_content,
                metadata={k: v for k, v in d.metadata.items() if v not in ("",None)}
                )
                for d in chunks
            ]
            logger.info(f"Enriquecendo os chunks com metadados")

            return enriched
            
        except Exception as e:
            logger.error(f"Falha ao dividir documentos: {str(e)}")
            raise
    
    def ingest_to_database(self, documents: List[Document], collection_name: str = "pdf_documents"):
        """Inclui documentos no banco de dados PGVector"""
        try:
            logger.info(f"Incluindo {len(documents)} documentos no banco de dados...")
            
            # Cria instância do PGVector
            vectorstore = PGVector(
                collection_name=collection_name,
                connection=self.connection_string,
                embeddings=self.embeddings,
                use_jsonb=True
            )
            
            # Adiciona documentos com rastreamento de progresso
            console.print("[bold blue]Incluindo documentos...[/bold blue]")
            
            # Processa documentos em lotes para evitar problemas de memória
            batch_size = 50
            for i in track(range(0, len(documents), batch_size), description="Processando lotes..."):
                batch = documents[i:i + batch_size]
                vectorstore.add_documents(batch)
                logger.info(f"Lote processado {i//batch_size + 1}")
            
            logger.info("Documentos incluidos com sucesso!")
            console.print("[bold green]✓ Inclusão concluída com sucesso![/bold green]")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Falha ao incluir documentos: {str(e)}")
            console.print(f"[bold red]✗ Ingestão falhou: {str(e)}[/bold red]")
            raise
    
    def process_pdf(self, pdf_path: str, collection_name: str = "pdf_documents") -> PGVector:
        """Pipeline completo de processamento de PDF"""
        try:
            console.print(f"[bold cyan]Iniciando ingestão de PDF: {pdf_path}[/bold cyan]")
            
            # Etapa 1: Carregar PDF
            documents = self.load_pdf(pdf_path)
            
            # Etapa 2: Dividir em chunks
            chunks = self.split_documents(documents)
            
            # Etapa 3: Adicionar metadados
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "source_file": os.path.basename(pdf_path),
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                })
            
            # Etapa 4: Ingerir no banco de dados
            vectorstore = self.ingest_to_database(chunks, collection_name)
            
            console.print("[bold green]✓ Processamento de PDF concluído![/bold green]")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Processamento de PDF falhou: {str(e)}")
            console.print(f"[bold red]✗ Processamento de PDF falhou: {str(e)}[/bold red]")
            raise


def ingest_pdf():
    """Função principal para executar ingestão de PDF"""
    try:
        # Pega o caminho do PDF dos argumentos ou do .env
        if len(sys.argv) > 2:
            console.print("[bold red]Uso: python ingest.py [caminho_para_pdf][/bold red]")
            sys.exit(1)

        if len(sys.argv) == 2:
            pdf_path = sys.argv[1]
        else:
            pdf_path = os.getenv("PDF_PATH")
            if not pdf_path:
                console.print("[bold red]Erro: Forneça o caminho do PDF como argumento ou defina PDF_PATH no .env[/bold red]")
                sys.exit(1)
        
        # Valida caminho do PDF
        if not pdf_path.endswith('.pdf'):
            console.print("[bold red]Por favor, forneça um arquivo PDF válido[/bold red]")
            sys.exit(1)
        
        # Inicializa ingestor
        ingestor = PDFIngestor()

        # Pega a Collection 
        collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME")
        if not collection_name:
             collection_name = Path(pdf_path).stem  # Usa nome do arquivo como nome da coleção
        
        # Processa PDF
        ingestor.process_pdf(pdf_path, collection_name)
        
        console.print(f"[bold green]PDF incluido com sucesso na coleção: {collection_name}[/bold green]")
        
    except KeyboardInterrupt:
        console.print("[bold yellow]Ingestão cancelada pelo usuário[/bold yellow]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Execução principal falhou: {str(e)}")
        console.print(f"[bold red]Ingestão falhou: {str(e)}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    ingest_pdf()



'''
Exemplo de Saida :

2025-09-19 11:24:38,752 - INFO - PDFIngestor inicializado com sucesso
Iniciando ingestão de PDF: ./document.pdf
2025-09-19 11:24:38,754 - INFO - Carregando PDF: ./document.pdf
2025-09-19 11:24:39,066 - INFO - Carregadas 34 páginas do PDF
2025-09-19 11:24:39,066 - INFO - Dividindo documentos em chunks...
2025-09-19 11:24:39,067 - INFO - Criados 67 chunks
2025-09-19 11:24:39,067 - INFO - Incluindo 67 documentos no banco de dados...
Incluindo documentos...
Processando lotes... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% -:--:--2025-09-19 11:24:40,994 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
Processando lotes... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% -:--:--2025-09-19 11:24:41,840 - INFO - Lote processado 1
Processando lotes... ━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━  50% -:--:--2025-09-19 11:24:43,445 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
Processando lotes... ━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━  50% -:--:--2025-09-19 11:24:44,036 - INFO - Lote processado 2
Processando lotes... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:04
2025-09-19 11:24:44,038 - INFO - Documentos incluidos com sucesso!
✓ Inclusão concluída com sucesso!
✓ Processamento de PDF concluído!
PDF incluido com sucesso na coleção: fullcycle_langchain
'''