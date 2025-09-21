# Desafio MBA Engenharia de Software com IA - Full Cycle

Visando apreender e transformar todo conhecimento passado pela FullCyle. 
Nesse projeto irei usar os seguintes tópicos abordados na modulo de nivelamento :

- [x] API Key OpenAi e GenAi 
- [x] Chamando a LLM pela primeira vez 
- [x] Prompt Templates 
- [x] Iniciando com Chains  
- [x] Criando Pipeline com mais etapas 
- [x] Iniciando com sumarização
- [x] TextSpliter e Sumarização
- [x] Criando pipeline customizado de sumarização
- [x] Agentes e ReAct
- [x] Criando Tools / Desenvolvendo Agentes
- [x] Introduçao a Data Loading e RAG 
- [x] Carregamento de PDF 
- [x] Iniciando com PgVector
- [x] Enriquecendo documentos
- [x] Criando Store para PGVector
- [x] Fazendo a ingestão dos documentos
- [x] Fazendo busca vetorial

> Todo o projeto foi feito para OpenAi conforme nivelamento.

# PDF RAG Chatbot with LangChain and PGVector

Este projeto implementa um sistema de RAG (Retrieval-Augmented Generation) para fazer perguntas sobre documentos PDF usando LangChain, PGVector e OpenAI.

## 🚀 Funcionalidades

- **Ingestão de PDF**: Carrega, divide em chunks e armazena embeddings no PostgreSQL com PGVector
- **Busca por Similaridade**: Tool personalizada para busca semântica nos documentos
- **Sumarização Inteligente**: Sumariza o contexto recuperado antes de enviá-lo para o LLM
- **Agente ReAct**: Utiliza padrão ReAct para raciocínio e ação com ferramentas
- **Interface Interativa**: Chat em linha de comando com interface rica (vou usar claude)
- **Logs Detalhados**: Sistema completo de logging para debugging (vou usar claude)

## 📋 Pré-requisitos

- Python 3.8+
- Docker e Docker Compose
- Chave da API OpenAI

## 🛠️ Instalação

1. **Clone o repositório e instale dependências:**
```bash
pip install -r requirements.txt
```

2. **Configure as variáveis de ambiente:**
Edite o arquivo `.env` e adicione sua chave OpenAI:
```bash
OPENAI_API_KEY=sk-your-api-key-here
```

3. **Inicie o banco PostgreSQL:**
```bash
docker compose up -d
```

Aguarde alguns segundos para o banco inicializar completamente.

## 🔄 Uso

### 1. Ingestão de PDF

Edite o arquivo `.env` e adicione o caminho do PDF
que ira buscar como default ou faça via script.

```bash
PDF_PATH=caminho-para-seu-documento.pdf
```

Para fazer a ingestão de um documento PDF:

```bash
python ingest.py caminho/para/seu/documento.pdf
```

Exemplo:
```bash
python ingest.py example.pdf
```

O script irá:
- Carregar e dividir o PDF em chunks
- Criar embeddings usando OpenAI
- Armazenar no PostgreSQL com PGVector
- Mostrar progresso em tempo real

### 2. Chat Interativo

Para iniciar o chat com o documento:

```bash
python chat.py
```

### 3. Exemplo de Uso Completo

```bash
# 1. Suba o banco
docker compose up -d

# 2. Faça a ingestão
python ingest.py meu_documento.pdf

# 3. Inicie o chat
python chat.py meu_documento

# 4. Faça perguntas sobre o PDF
Your question: Qual é o tema principal do documento?
```

## 🏗️ Arquitetura

### Componentes Principais

1. **ingest.py**
   - Carregamento de PDF com PyPDFLoader
   - Divisão em chunks com RecursiveCharacterTextSplitter
   - Criação de embeddings com OpenAI
   - Armazenamento em PGVector

2. **search.py**
   - Tool de busca por similaridade
   - Sistema de sumarização com chains
   - Pipeline usando pipes do LangChain

3. **chat.py**
   - Agente ReAct com ferramentas customizadas
   - Interface interativa com Rich
   - Memória de conversação (talvez veremos)
   - Sistema robusto de tratamento de erros

### Fluxo de Dados

```
PDF → Chunks → Embeddings → PGVector
                                ↓
User Question → ReAct Agent → Search Tool → Similarity Search
                                ↓
Retrieved Docs → Summarization → Context → LLM → Answer
```

## 🔧 Configuração Avançada

### Variáveis de Ambiente (.env)

```bash
# OpenAI
OPENAI_API_KEY=sua_chave_aqui
EMBEDDING_MODEL=text-embedding-3-small  # ou text-embedding-ada-002
CHAT_MODEL=gpt-5o-mini                 

# Database
DATABASE_URL=postgresql+psycopg://postgres:postgres@host.docker.internal:5432/rag

#Collection
PG_VECTOR_COLLECTION_NAME=fullcycle_langchain

#Processamento PDF 
PDF_PATH=./document.pdf
CHUNK_SIZE=1000
CHUNK_OVERLAP=150

#Logs
LOG_LEVEL=INFO

```

### Personalizações

#### Modificar o Tamanho dos Chunks
```python
# No ingest.py, ajuste:
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Chunks maiores
    chunk_overlap=300,
    separators=["\n\n", "\n", " ", ""]
)
```

#### Customizar o Prompt ReAct
```python
# No chat.py, modifique o template do react_prompt
react_prompt = PromptTemplate(
    input_variables=["tools", "tool_names", "input", "agent_scratchpad", "chat_history"],
    template="""Seu prompt customizado aqui..."""
)

# Levei horas para descobrir que não da para traduzir o prompt ReAct do Langchain
# temos que utilizar as keywords em inglês além do pulo do gato que foi apenas a linhas abaixo :
# IMPORTANTE: No campo Action, use apenas o nome da ferramenta SEM colchetes. Exemplo: "document_search" e NÃO "[document_search]"

```

## 🐛 Solução de Problemas

### Erro de Conexão com Banco
```bash
# Verifique se o PostgreSQL está rodando
docker compose ps

# Reinicie se necessário
docker compose restart
```

### Erro de API OpenAI
- Verifique se a chave da API está correta no `.env`
- Confirme se você tem créditos disponíveis
- Teste a conectividade:
```python
from openai import OpenAI
client = OpenAI(api_key="sua_chave")
```

## 📁 Estrutura do Projeto

```
.
├── docker-compose.yml       # Configuração PostgreSQL + PGVector
├── requirements.txt         # Dependências Python
├── .env                     # Variáveis de ambiente
├── README.md                # Esta documentação
├── src
|   ├── ingest.py            # Script de ingestão de PDFs
|   ├── search.py            # Tools de busca e sumarização
|___├── chat.py              # Interface de chat interativo

```

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

Para mais informações, consulte a [documentação oficial do LangChain](https://docs.langchain.com/).