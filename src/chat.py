"""
Interface de Chat Interativo
Implementa agente ReAct para responder perguntas com contexto de documentos PDF
"""

import os
import sys
import logging
from typing import List

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

from search import create_search_tool


PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

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
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


# Carrega variáveis de ambiente
load_dotenv()

# Configuração de logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console = Console()


class PDFChatbot:
    """Chatbot interativo para perguntas e respostas sobre documentos PDF usando agente ReAct"""
    
    def __init__(self, collection_name: str = "pdf_documents"):
        """Inicializa o chatbot com agente ReAct"""
        try:
            self.collection_name = collection_name
            console.print(f"[bold blue]Inicializando PDF Chatbot para coleção: {collection_name}[/bold blue]")
            
            # Inicializa LLM
            self.llm = ChatOpenAI(
                model=os.getenv('CHAT_MODEL', 'gpt-5o-mini'),
                api_key=os.getenv('OPENAI_API_KEY'),
                temperature=0.1,
                streaming=True
            )
            
            # Memória desabilitada temporariamente para simplificar debugging
            self.memory = None
            
            # Cria ferramentas
            self.tools = self._create_tools()

            # Cria agente ReAct
            self.agent = self._create_react_agent()
            
            # Cria executor do agente
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3
            )
            
            logger.info("PDFChatbot inicializado com sucesso")
            console.print("[bold green]✓ Chatbot inicializado com sucesso![/bold green]")
            
        except Exception as e:
            logger.error(f"Falha ao inicializar PDFChatbot: {str(e)}")
            console.print(f"[bold red]✗ Falha na inicialização do Chatbot: {str(e)}[/bold red]")
            raise
    
    def _create_tools(self) -> List[Tool]:
        """Cria ferramentas para o agente ReAct"""
        try:            
            # Ferramenta de busca de documentos
            search_tool = create_search_tool(self.collection_name)

            tools = [search_tool]
                                    
            logger.info(f"Criadas {len(tools)} ferramentas para o agente")
            return tools
            
        except Exception as e:
            logger.error(f"Falha ao criar ferramentas: {str(e)}")
            raise
    
    def _create_react_agent(self):
        """Cria agente ReAct com prompt personalizado"""
        try:
            """
                Pulo do Gato : 

                    Estrutura : 

                        Seu prompt ....

                        Prompt ReAct ... : Sem traduzir as Keywords , pos langchain precisa delas 

                        Caso utilize tools adicione a linha : 

                        # IMPORTANTE: No campo Action, use apenas o nome da ferramenta SEM colchetes. Exemplo: "document_search" e NÃO "[document_search]"

                        Begin!

                        Question: {input}
                        Thought:{agent_scratchpad}
            """
            # Template de prompt ReAct integrando regras do PROMPT_TEMPLATE em português
            react_prompt = PromptTemplate(
                input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
                template="""Responda as perguntas da melhor forma possível. Você tem acesso às seguintes ferramentas:

{tools}

REGRAS FUNDAMENTAIS:
- Responda somente com base no CONTEXTO obtido pelas ferramentas.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.
- SEMPRE responda em português, mesmo que a ferramenta retorne em inglês.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Use o seguinte formato EXATO:

Question: a pergunta de entrada que você deve responder
Thought: você deve sempre pensar sobre o que fazer
Action: a ação a ser tomada, deve ser uma das seguintes: [{tool_names}]
Action Input: a entrada para a ação
Observation: o resultado da ação
... (este ciclo Thought/Action/Action Input/Observation pode se repetir quantas vezes necessário)
Thought: agora eu sei a resposta final
Final Answer: [Aplique as REGRAS FUNDAMENTAIS aqui e responda SEMPRE em português]

IMPORTANTE: No campo Action, use apenas o nome da ferramenta SEM colchetes. Exemplo: "document_search" e NÃO "[document_search]"

Comece!

Question: {input}
Thought:{agent_scratchpad}"""
            )
            
            # Cria agente ReAct
            agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=react_prompt
            )

            logger.info("Agente ReAct criado com sucesso")
            return agent
            
        except Exception as e:
            logger.error(f"Falha ao criar agente ReAct: {str(e)}")
            raise
    
    def ask_question(self, question: str) -> str:
        """Faz uma pergunta ao chatbot"""
        try:
            logger.info(f"Processando pergunta: '{question}'")
            
            console.print(f"\n[bold cyan]🤔 Processando sua pergunta...[/bold cyan]")
            
            # Usa executor do agente para obter resposta
            response = self.agent_executor.invoke({"input": question})
            
            answer = response.get("output", "Não consegui gerar uma resposta.")
            
            logger.info("Pergunta processada com sucesso")
            return answer
            
        except Exception as e:
            error_msg = f"Erro ao processar pergunta: {str(e)}"
            logger.error(error_msg)
            console.print(f"[bold red]✗ {error_msg}[/bold red]")
            return error_msg
    
    def start_chat(self):
        """Inicia sessão de chat interativo"""
        try:
            console.print(Panel.fit(
                "[bold green]Chatbot de Documentos PDF[/bold green]\n"
                f"Coleção: {self.collection_name}\n"
                "Digite suas perguntas sobre o conteúdo do PDF.\n"
                "Digite 'quit', 'exit', ou 'bye' para encerrar a conversa.",
                title="Bem-vindo"
            ))
            
            while True:
                try:
                    # Obtém entrada do usuário
                    question = Prompt.ask("\n[bold blue]Sua pergunta[/bold blue]")
                    
                    # Verifica comandos de saída
                    if question.lower() in ['quit', 'exit', 'bye', 'q']:
                        console.print("[bold yellow]Tchau! 👋[/bold yellow]")
                        break
                    
                    if not question.strip():
                        console.print("[bold yellow]Por favor digite uma pergunta.[/bold yellow]")
                        continue
                    
                    # Obtém resposta do chatbot
                    answer = self.ask_question(question)
                    
                    # Exibe resposta
                    console.print("\n[bold green]🤖 Resposta:[/bold green]")
                    console.print(Panel(Markdown(answer), title="Resposta", border_style="green"))
                    
                except KeyboardInterrupt:
                    console.print("\n[bold yellow]Chat interrompido pelo usuário.[/bold yellow]")
                    break
                except Exception as e:
                    logger.error(f"Erro no chat: {str(e)}")
                    console.print(f"[bold red]Erro no chat: {str(e)}[/bold red]")
                    
        except Exception as e:
            logger.error(f"A Sessão do chat falhou: {str(e)}")
            console.print(f"[bold red] A Sessão do chat falhou: {str(e)}[/bold red]")


def main():
    """Função principal para iniciar a interface de chat"""
    try:
        # Obtém nome da coleção da linha de comando ou usa padrão
        collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME")
        if len(sys.argv) > 1:
            collection_name = sys.argv[1]
        
        console.print(f"[bold cyan]Iniciando PDF Chatbot...[/bold cyan]")
        
        # Inicializa chatbot
        chatbot = PDFChatbot(collection_name)
        
        # Inicia sessão de chat
        chatbot.start_chat()
        
    except KeyboardInterrupt:
        console.print("[bold yellow]Inicialização cancelada pelo usuário[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Execução principal falhou: {str(e)}")
        console.print(f"[bold red]Falha na inicialização: {str(e)}[/bold red]")
        
        # Mostra mensagens de erro úteis
        if "OPENAI_API_KEY" in str(e):
            console.print("[bold yellow]💡 Certifique-se de que sua chave da API OpenAI está configurada no arquivo .env[/bold yellow]")
        elif "DATABASE_URL" in str(e):
            console.print("[bold yellow]💡 Certifique-se de que o banco PostgreSQL está executando (docker compose up -d)[/bold yellow]")
        elif "collection" in str(e).lower():
            console.print("[bold yellow]💡 Certifique-se de ter ingerido um PDF primeiro usando: python ingest.py <arquivo_pdf>[/bold yellow]")
        
        sys.exit(1)


if __name__ == "__main__":
    main()


""""
Exemplo : 

Iniciando PDF Chatbot...
Inicializando PDF Chatbot para coleção: fullcycle_langchain
2025-09-21 18:31:05,259 - INFO - Configuração da chain de sumarização concluída
2025-09-21 18:31:05,259 - INFO - DocumentSearchTool inicializada para coleção: fullcycle_langchain
2025-09-21 18:31:05,261 - INFO - Ferramenta de busca criada com sucesso
2025-09-21 18:31:05,261 - INFO - Criadas 1 ferramentas para o agente
2025-09-21 18:31:05,262 - INFO - Agente ReAct criado com sucesso
2025-09-21 18:31:05,262 - INFO - PDFChatbot inicializado com sucesso
✓ Chatbot inicializado com sucesso!
╭──────────────────────── Bem-vindo ────────────────────────╮
│ Chatbot de Documentos PDF                                 │
│ Coleção: fullcycle_langchain                              │
│ Digite suas perguntas sobre o conteúdo do PDF.            │
│ Digite 'quit', 'exit', ou 'bye' para encerrar a conversa. │
╰───────────────────────────────────────────────────────────╯

Sua pergunta: Qual o faturamento da empresa SuperTechIABrazil?
2025-09-21 18:31:07,491 - INFO - Processando pergunta: 'Qual o faturamento da empresa SuperTechIABrazil?'

🤔 Processando sua pergunta...


> Entering new AgentExecutor chain...
2025-09-21 18:31:08,253 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Action: document_search  
Action Input: "faturamento da empresa SuperTechIABrazil"  2025-09-21 18:31:08,624 - INFO - Ferramenta executando busca para: 'faturamento da empresa SuperTechIABrazil'
2025-09-21 18:31:08,624 - INFO - Executando busca por similaridade para: 'faturamento da empresa SuperTechIABrazil' (k=10)
2025-09-21 18:31:09,192 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
2025-09-21 18:31:09,222 - INFO - Encontrados 10 documentos similares
2025-09-21 18:31:09,223 - INFO - Sumarizando 10 documentos
2025-09-21 18:31:11,424 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-09-21 18:31:11,436 - INFO - Sumarização de contexto concluída
2025-09-21 18:31:11,437 - INFO - Busca da ferramenta concluída com sucesso
2025-09-21 18:31:11,501 - INFO - Retrying request to /chat/completions in 0.419732 seconds
2025-09-21 18:31:12,876 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
The company SuperTechIABrazil has a reported faturamento (revenue) of R$ 10.000.000,00 for the year 2025. This information is found in Document 3 and Document 4. No additional details about the company's financial performance or context are provided in the other document chunks.Agora eu sei a resposta final  
Final Answer: O faturamento da empresa SuperTechIABrazil é de R$ 10.000.000,00 para o ano de 2025.

> Finished chain.
2025-09-21 18:31:13,653 - INFO - Pergunta processada com sucesso

🤖 Resposta:
╭─────────────────────────────────────────────────────────────────────────────── Resposta ───────────────────────────────────────────────────────────────────────────────╮
│ O faturamento da empresa SuperTechIABrazil é de R$ 10.000.000,00 para o ano de 2025.                                                                                   │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Sua pergunta: Qual o cor do ceu ?                                 
2025-09-21 18:38:53,855 - INFO - Processando pergunta: 'Qual o cor do ceu ?'

🤔 Processando sua pergunta...


> Entering new AgentExecutor chain...
2025-09-21 18:38:54,882 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2025-09-21 18:38:56,420 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
Não tenho informações necessárias para responder sua pergunta.Invalid Format: Missing 'Action:' after 'Thought:'Question: Qual a cor do céu?
Thought: Não tenho informações necessárias para responder sua pergunta.
Final Answer: Não tenho informações necessárias para responder sua pergunta.

> Finished chain.
2025-09-21 18:38:57,001 - INFO - Pergunta processada com sucesso

🤖 Resposta:
╭─────────────────────────────────────────────────────────────────────────────── Resposta ───────────────────────────────────────────────────────────────────────────────╮
│ Não tenho informações necessárias para responder sua pergunta.                                                                                                         │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

"""