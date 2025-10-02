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

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import  AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

from search import create_search_tool


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
                model=os.getenv('CHAT_MODEL', 'gpt-5-nano'),
                api_key=os.getenv('OPENAI_API_KEY'),
                temperature=0.1,
                disable_streaming=True            
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
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=3
            )
            
            logger.info("PDFChatbot inicializado com sucesso")
            console.print("[bold green]✓ Chatbot inicializado com sucesso![/bold green]")
            
        except Exception as e:
            logger.error(f"Falha ao inicializar PDFChatbot: {str(e)}")
            console.print(f"[bold red]✗ Falha na inicialização do Chatbot: {str(e)}[/bold red]")
            raise
    
    def _create_tools(self) -> List[tool]:
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
            # Template de prompt ReAct integrando regras do PROMPT_TEMPLATE em português
            react_prompt = PromptTemplate.from_template(
            """                
                Answer the following questions as best you can. You have access to the following tools.
                Only use the information you get from the tools, even if you know the answer.

                {tools}

                Use the following format:

                Question: the input question you must answer
                Thought: you should always think about what to do
                Action: the action to take, should be one of [{tool_names}]
                Action Input: the input to the action
                Observation: the result of the action

                ... (this Thought/Action/Action Input/Observation can repeat N times)
                Thought: I now know the final answer
                Final Answer: the final answer to the original input question

                Rules:

                - If you choose an Action, do NOT include Final Answer in the same step.
                - After Action and Action Input, stop and wait for Observation.
                - Never search the internet. Only use the tools provided.

                Begin!
                Question: {input}
                Thought: {agent_scratchpad}
            """
            )
            
            # Cria agente ReAct
            agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=react_prompt,
                stop_sequence=False
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

Inicializando PDF Chatbot para coleção: fullcycle_langchain
✓ Chatbot inicializado com sucesso!
╭──────────────────────── Bem-vindo ────────────────────────╮
│ Chatbot de Documentos PDF                                 │
│ Coleção: fullcycle_langchain                              │
│ Digite suas perguntas sobre o conteúdo do PDF.            │
│ Digite 'quit', 'exit', ou 'bye' para encerrar a conversa. │
╰───────────────────────────────────────────────────────────╯

Sua pergunta: Qual a cor do ceu ?

🤔 Processando sua pergunta...

🤖 Resposta:
╭─────────────────────────────────────────────────────────────────────────────── Resposta ───────────────────────────────────────────────────────────────────────────────╮
│ Não tenho informações necessárias para responder sua pergunta.                                                                                                         │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Sua pergunta: Qual o faturamento da empresa SuperTechIABrazil?

🤔 Processando sua pergunta...

🤖 Resposta:
╭─────────────────────────────────────────────────────────────────────────────── Resposta ───────────────────────────────────────────────────────────────────────────────╮
│ SuperTechIABrazil faturamento: R$ 10.000.000,00 (dez milhões de reais) no ano de 2025. Aparece nos Documentos 1 e 2 (Chunk 5) com o mesmo valor/ano. Não há outros     │
│ valores de faturamento para a SuperTechIABrazil nos trechos fornecidos.                                                                                                │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Sua pergunta: 
"""