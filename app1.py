from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from run import SentimentAnalysis  # Assuming this is the class that does your analysis
import uvicorn

app = FastAPI()

# Initialize the SentimentAnalysis instance
sentiments = SentimentAnalysis()

class AnalysisRequest(BaseModel):
    quote: str
    num_of_posts: Optional[int] = 10  # default to 10 if not provided

@app.post("/sentiment_analysis")
async def analyze_sentiment(request: AnalysisRequest):
    output = None
    status = True
    reason = ""

    try:
        output = sentiments.analysis(request.quote, request.num_of_posts)
        status = False
    except Exception as e:
        reason = "Exception " + str(e)
    
    if status == False:
        message = "Success"
    else:
        message = "Failure"
    return   {
        'error': status,
        'message': message,
        'reason': reason,
        'status': 200,
        'response_data': output,
    }


if __name__ == "__main__":
    uvicorn.run("app1:app", host="127.0.0.1", port=8080, reload=True)