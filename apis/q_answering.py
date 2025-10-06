from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import requests
import spacy
import oracledb
# from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
# from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import logging
import re
from datetime import datetime
import json
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class YouTubeVideo(BaseModel):
    id: str
    identifier: str
    query: str
    videoId: str
    title: str
    description: str
    publishedAt: str
    channelTitle: str
    thumbnailUrl: str
    duration: str
    created: str

class QuestionAnswer(BaseModel):
    videoId: str
    title: str
    publishedAt: str
    channelTitle: str
    question: str
    answer: str

class NewsQuestionAnswer(BaseModel):
    title: str
    description: str
    entities: str
    sentiments: str
    question: str
    answer: str
    published_at: str

class QAResponse(BaseModel):
    success: bool
    processed_videos: int
    qa_pairs: List[QuestionAnswer]
    message: str

class NewsQAResponse(BaseModel):
    success: bool
    processed_news: int
    qa_pairs: List[NewsQuestionAnswer]
    message: str

# Global variables for models
nlp = None
llm_chain = None
"""You are a precise assistant.
Answer the following question based only on the provided context.

Context:
{context}

Question:
{question}

Instructions:
- Be concise (max 3 sentences).
- Prioritize precision and facts.
- Provide reasoning only if it improves clarity, not speculation.
- Do NOT add unnecessary detail.

Answer:"""
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    global nlp, llm_chain
    logger.info("Loading SpaCy model...")
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        logger.error("SpaCy English model not found. Please install it using: python -m spacy download en_core_web_sm")
        raise
    
    logger.info("Initializing Ollama LLM...")
    try:
        # Instantiate the ChatOllama model, passing the options dictionary
        llm = ChatOllama(
            model="phi3:latest",
            num_keep=-1,
            seed=42,
            num_predict=-1,
            top_k=20,
            top_p=0.9,
            num_thread=2,
            num_gpu=-1
        )
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are an AI assistant writing in the style of a professional journalist. 

Guidelines:
1. Base your answer strictly on the provided context.
2. Present the information in a clear, neutral, and factual way, avoiding personal opinions.
3. If the context does not contain the answer, respond with: 
   "The available context does not provide enough information to answer this question."
4. Write in the tone of a journalist: precise, objective, and informative.
5. Use short, well-structured sentences and avoid unnecessary embellishments.
6. Be concise (max 3 sentences).

Context:
{context}

Question:
{question}

Journalistic Answer:
"""
        )
        # prompt = PromptTemplate.from_template(prompt_template)
        llm_chain = prompt_template | llm
    except Exception as e:
        logger.error(f"Failed to initialize Ollama: {e}")
        raise
    
    yield
    # Cleanup on shutdown
    logger.info("Shutting down...")

app = FastAPI(
    title="YouTube QA System API",
    description="API for YouTube search, question generation, and Q&A using AI",
    version="1.0.0",
    lifespan=lifespan
)

class DatabaseManager:
    def __init__(self):
        self.connection = None
    
    def connect(self):
        """Connect to Oracle Database"""
        try:
            # Update these connection parameters according to your Oracle DB setup
            dsn = "localhost:1521/FREE"  # Update with your Oracle DB details
            self.connection = oracledb.connect(
                user="SYS",
                password="oracle",
                dsn=dsn,
                mode=oracledb.SYSDBA
            )
            logger.info("Connected to Oracle Database")
        except Exception as e:
            logger.error(f"Failed to connect to Oracle DB: {e}")
            raise
    
    def create_table_if_not_exists(self):
        """Create the QA table if it doesn't exist"""
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        try:
            cursor.execute("""
                CREATE TABLE youtube_qa (
                    id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    videoId VARCHAR2(50) NOT NULL,
                    title VARCHAR2(500) NOT NULL,
                    publishedAt VARCHAR2(50) NOT NULL,
                    channelTitle VARCHAR2(200) NOT NULL,
                    question CLOB NOT NULL,
                    answer CLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.connection.commit()
            logger.info("Created youtube_qa table")
        except oracledb.DatabaseError as e:
            if "ORA-00955" in str(e):  # Table already exists
                logger.info("youtube_qa table already exists")
            else:
                logger.error(f"Error creating table: {e}")
                raise
        finally:
            cursor.close()
    # class NewsQuestionAnswer(BaseModel):
    #     title: str
    #     description: str
    #     entities: str
    #     question: str
    #     answer: str
    #     published_at: str
    #     created_at: str

    def insert_news_clusters_qa(self, qa_data: NewsQuestionAnswer):
        """Insert Q&A pair into database"""
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        try:
            cursor.execute("""
                INSERT INTO news_qa (title, description, entities, sentiments, question, answer, published_at)
                VALUES (:1, :2, :3, :4, :5, :6, :7)
            """, (
                qa_data.title,
                qa_data.description,
                qa_data.entities,
                qa_data.sentiments,
                qa_data.question,
                qa_data.answer,
                qa_data.published_at,                
            ))
            self.connection.commit()
            logger.info(f"Inserted Q&A for news cluster: {qa_data.title}")
        except Exception as e:
            logger.error(f"Failed to insert Q&A: {e}")
            raise
        finally:
            cursor.close()
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
    
    def insert_qa_pair(self, qa_data: QuestionAnswer):
        """Insert Q&A pair into database"""
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        try:
            cursor.execute("""
                INSERT INTO youtube_qa (videoId, title, publishedAt, channelTitle, question, answer)
                VALUES (:1, :2, :3, :4, :5, :6)
            """, (
                qa_data.videoId,
                qa_data.title,
                qa_data.publishedAt,
                qa_data.channelTitle,
                qa_data.question,
                qa_data.answer
            ))
            self.connection.commit()
            logger.info(f"Inserted Q&A for video: {qa_data.videoId}")
        except Exception as e:
            logger.error(f"Failed to insert Q&A: {e}")
            raise
        finally:
            cursor.close()
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()

class QuestionGenerator:
    """Generate intelligent questions from text using SpaCy"""
    
    @staticmethod
    def extract_entities_and_topics(text: str) -> Dict[str, List[str]]:
        """Extract named entities and key topics from text"""
        doc = nlp(text)
        
        entities = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],  # Geopolitical entities
            "EVENT": [],
            "PRODUCT": [],
            "MONEY": [],
            "DATE": []
        }
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_] = list(set(entities[ent.label_] + [ent.text]))
        
        # Extract noun phrases as topics
        noun_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 4]
        
        return {
            "entities": entities,
            "topics": noun_phrases[:10]  # Limit to top 10 topics
        }
    
    @staticmethod
    def generate_questions(description: str, title: str) -> List[str]:
        """Generate smart questions based on video description and title"""
        if not description or len(description.strip()) < 10:
            return [f"What is this video about: {title}?"]
        
        questions = []
        
        # Extract key information
        extracted = QuestionGenerator.extract_entities_and_topics(description)
        entities = extracted["entities"]
        topics = extracted["topics"]
        
        # Generate entity-based questions
        if entities["PERSON"]:
            persons = list(set(entities["PERSON"][:3]))  # Max 3 unique persons
            for person in persons:
                questions.append(f"Who is {person} and what is their role in this video?")
        
        if entities["ORG"]:
            orgs = list(set(entities["ORG"][:2]))
            for org in orgs:
                questions.append(f"What is {org} and how is it related to the video content?")
        
        if entities["GPE"]:
            places = list(set(entities["GPE"][:2]))
            for place in places:
                questions.append(f"What happens in {place} according to this video?")
        
        if entities["DATE"]:
            dates = list(set(entities["DATE"][:2]))
            for date in dates:
                questions.append(f"What is significant about {date} in this video?")
        
        # Generate topic-based questions
        for topic in topics[:3]:  # Max 3 topics
            if len(topic.split()) > 1:  # Only multi-word topics
                questions.append(f"Can you explain more about {topic}?")
        
        # Generate general questions
        questions.extend([
            f"What is the main topic of the video '{title}'?",
            "What are the key points discussed in this video?",
            "What can viewers learn from this video?"
        ])
        
        # Remove duplicates and limit to 5 questions
        unique_questions = list(dict.fromkeys(questions))[:5]
        
        return unique_questions if unique_questions else [f"What is this video about: {title}?"]

# Initialize database manager
db_manager = DatabaseManager()

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    try:
        db_manager.create_table_if_not_exists()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

@app.get("/")
async def root():
    return {"message": "YouTube QA System API", "status": "running"}


@app.get("/youtube_search_by_query")
async def youtube_search_by_query(query: str = Query(..., description="Search query for YouTube videos")):
    """Fetch YouTube videos and return basic information"""
    try:
        # Make request to the YouTube search service
        youtube_api_url = f"http://10.42.0.243:5000/youtube_search_by_query"
        params = {"query": str(query)}
        
        response = requests.get(youtube_api_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to YouTubeVideo objects
        videos = []
        for item in data:
            item["id"] = str(item.get("id"))  # force string
            video = YouTubeVideo(**item)
            videos.append(video)
        
        return {
            "success": True,
            "query": query,
            "count": len(videos),
            "videos": videos
        }
    
    except requests.RequestException as e:
        logger.error(f"Failed to fetch YouTube data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch YouTube data: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
# class NewsQAResponse(BaseModel):
#     success: bool
#     processed_cluster: int
#     qa_pairs: List[NewsQuestionAnswer]
#     message: str
# date: str = Query(..., description="Search news clusters by date")

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import html
import re
import oracledb
import spacy
from database.SelectTables import SelectTables

def clean_html(text):
    return BeautifulSoup(text or "", "html.parser").get_text()

def sanitize_text(text: str) -> str:
    text = clean_html(text)
    text = html.unescape(text)
    text = re.sub(r'[<>]', '', text)
    return text.strip()


@app.post("/filter_news_qa_table_by_ids")
async def filter_news_qa_table_by_ids(fth_id: int, snd_id: int):
    try:
        data = SelectTables().filter_news_qa_table_by_ids(fth_id, snd_id)
        return data
    except Exception as e:
        logger.error(f"Failed to process news qa: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process news qa: {str(e)}") 

def news_by_ids_interval_for_qa(fth_id: int, snd_id: int) -> list:
    try:
        sentiment = SentimentIntensityAnalyzer()
        nlp = spacy.load("en_core_web_md")
        data = SelectTables().filter_news_api_table_by_ids(fth_id, snd_id)
        results = []
        sentiments = []
        for row in data:
            source = row["source"]
            author = row["author"]
            title = row["title"]
            description = row["description"]
            url = row["url"]
            urlToImage = row["urltoimage"]
            publishedAt = row["publishedat"]
            content = row["content"]

            dictionary = sentiment.polarity_scores(str(description))
            positive = round(float(dictionary['pos']), 4)
            negative = round(float(dictionary['neg']), 4)
            neutral = round(float(dictionary['neu']), 4)
            compound = round(float(dictionary['compound']), 4)
            
            results.append({
                "source": source,
                "author": author,
                "title": title,
                "description": description,
                "url": url,
                "urlToImage": urlToImage,
                "publishedAt": publishedAt,
                "content": content,
                "sentiments": {
                    "pos": positive, "neg": negative, "neu": neutral, "compound": compound
                },
                "entities": [ent.text for ent in nlp(sanitize_text(description).replace('&', ' ')).ents]
            })
        return results
    except Exception as e:
        endpoint: str = "/news_by_ids_interval_for_qa"
        logging.error(f"Error: {e} | {endpoint} | at {datetime.now()}")
        return []



@app.post("/news_clusters_qa", response_model=NewsQAResponse)
async def news_clusters_qa(fth_id: int, snd_id: int):
    """Process News Clusters: generate questions and answers, store in database"""
    try:
        clusters = news_by_ids_interval_for_qa(fth_id=fth_id, snd_id=snd_id)
        
        qa_response: list[NewsQuestionAnswer] = []
        processed_count = 0
        total_clusters = len(clusters)

        for cluster in clusters:
            if not cluster:
                continue

            try:
                
                entities: list[str] = cluster["entities"]
                sentiments = cluster["sentiments"]
                sentiment_str = (
                    f"positive={sentiments['pos']}, "
                    f"negative={sentiments['neg']}, "
                    f"neutral={sentiments['neu']}, "
                    f"compound={sentiments['compound']}"
                )
                logger.debug(f"Sentiments: {sentiment_str}")
                pos = float(sentiments["pos"])
                neg = float(sentiments["neg"])
                if neg > pos:
                    entities_text = ", ".join(entities) if entities else "N/A"
                    question = f"How are the entities {entities_text} related to the news article?"
                    logger.debug(f"Question: {question}")

                    context = (
                        f"News Article title: {cluster['title']}\n"
                        f"News Article description: {cluster['description']}\n"
                    )
                    logger.debug(f"Context: {context}")

                    answer = llm_chain.invoke({"context": context, "question": question})

                    qa_data = NewsQuestionAnswer(
                        title=cluster["title"],
                        description=cluster["description"],
                        entities=entities_text,
                        sentiments=sentiment_str,
                        question=question,
                        answer=answer.content,
                        published_at=cluster["publishedAt"]
                    )

                    db_manager.insert_news_clusters_qa(qa_data)
                    qa_response.append(qa_data)

                    processed_count += 1
                    logger.info(f"Processed cluster {processed_count}/{total_clusters}: {qa_data.title}")

            except Exception as inner_e:
                logger.error(f"Failed to process cluster '{cluster['title']}': {inner_e}")
                continue
        
        return NewsQAResponse(
            success=True,
            processed_news=processed_count,
            qa_pairs=qa_response,
            message=f"Successfully processed {processed_count} of {total_clusters} clusters, generated {len(qa_response)} Q&A pairs"
        )

    except Exception as e:
        logger.error(f"Failed to process news clusters: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process news clusters: {str(e)}")


@app.post("/process_videos_qa")
async def process_videos_qa(query: str = Query(..., description="Search query for YouTube videos")):
    """Process YouTube videos: generate questions and answers, store in database"""
    try:
        # First, get YouTube videos
        youtube_response = await youtube_search_by_query(query)
        videos = youtube_response["videos"]
        
        if not videos:
            return QAResponse(
                success=True,
                processed_videos=0,
                qa_pairs=[],
                message="No videos found for the query"
            )
        
        qa_pairs = []
        processed_count = 0
        
        for video in videos:
            try:
                # Generate questions using SpaCy
                questions = QuestionGenerator.generate_questions(video.description, video.title)
                video: YouTubeVideo = video
                # Process each question with LangChain + Ollama
                for question in questions:
                    try:
                        # Create context from video information
                        context = f"""
                        Video Title: {video.title}
                        Channel: {video.channelTitle}
                        Published: {video.publishedAt}
                        Description: {video.description}...
                        """
                        
                        # Generate answer using LLM
                        answer = llm_chain.invoke({"context": context, "question": question})
                        
                        # video_id = str(video.videoId) if not isinstance(video.videoId, str) else video.videoId

                        # Create QA pair
                        qa_pair = QuestionAnswer(
                            videoId=str(video.videoId),
                            title=str(video.title),
                            publishedAt=str(video.publishedAt),
                            channelTitle=str(video.channelTitle),
                            question=str(question),
                            answer=str(answer.content).strip()
                        )
                        
                        # Store in database
                        db_manager.insert_qa_pair(qa_pair)
                        qa_pairs.append(qa_pair)
                        
                    except Exception as e:
                        logger.error(f"Failed to process question '{question}' for video {str(video.title)}: {e}")
                        continue
                
                processed_count += 1
                logger.info(f"Processed video {processed_count}/{len(videos)}: {video.title}")
                
            except Exception as e:
                logger.error(f"Failed to process video {str(video.videoId)}: {e}")
                continue
        
        return QAResponse(
            success=True,
            processed_videos=processed_count,
            qa_pairs=qa_pairs,
            message=f"Successfully processed {processed_count} videos and generated {len(qa_pairs)} Q&A pairs"
        )
        
    except Exception as e:
        logger.error(f"Failed to process videos: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process videos: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "spacy_loaded": nlp is not None,
        "llm_loaded": llm_chain is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    db_manager.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
