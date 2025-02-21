from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class Agent(BaseModel):
    """Base agent class with common properties"""
    name: str
    role: str
    description: str
    llm: Any = Field(description="Language model for the agent")
    
class PlannerAgent(Agent):
    """Agent responsible for breaking down problems and planning steps"""
    def __init__(self, llm):
        super().__init__(
            name="Planner",
            role="Strategic Planner",
            description="Breaks down complex problems into manageable steps",
            llm=llm
        )
        
    def plan(self, query: str, context: List[Dict[str, Any]] = None) -> str:
        if context:
            template = """You are a strategic planning agent. Your role is to break down complex problems into clear, manageable steps.
            
            Given the following context and query, create a step-by-step plan to answer the question.
            Each step should be clear and actionable.
            
            Context:
            {context}
            
            Query: {query}
            
            Plan:"""
            context_str = "\n\n".join([f"Context {i+1}:\n{item['content']}" for i, item in enumerate(context)])
        else:
            template = """You are a strategic planning agent. Your role is to break down complex problems into clear, manageable steps.
            
            Given the following query, create a step-by-step plan to answer the question.
            Each step should be clear and actionable.
            
            Query: {query}
            
            Plan:"""
            context_str = ""
            
        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(query=query, context=context_str)
        response = self.llm.invoke(messages)
        return response.content

class ResearchAgent(Agent):
    """Agent responsible for gathering and analyzing information"""
    vector_store: Any = Field(description="Vector store for searching")
    
    def __init__(self, llm, vector_store):
        super().__init__(
            name="Researcher",
            role="Information Gatherer",
            description="Gathers and analyzes relevant information from knowledge bases",
            llm=llm,
            vector_store=vector_store
        )
        
    def research(self, query: str, step: str) -> List[Dict[str, Any]]:
        # Query all collections
        pdf_results = self.vector_store.query_pdf_collection(query)
        repo_results = self.vector_store.query_repo_collection(query)
        
        # Combine results
        all_results = pdf_results + repo_results
        
        if not all_results:
            return []
            
        # Have LLM analyze and summarize findings
        template = """You are a research agent. Your role is to analyze information and extract relevant details.
        
        Given the following research step and context, summarize the key findings that are relevant to this step.
        
        Step: {step}
        
        Context:
        {context}
        
        Key Findings:"""
        
        context_str = "\n\n".join([f"Source {i+1}:\n{item['content']}" for i, item in enumerate(all_results)])
        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(step=step, context=context_str)
        response = self.llm.invoke(messages)
        
        return [{"content": response.content, "metadata": {"source": "Research Summary"}}]

class ReasoningAgent(Agent):
    """Agent responsible for logical reasoning and analysis"""
    def __init__(self, llm):
        super().__init__(
            name="Reasoner",
            role="Logic and Analysis",
            description="Applies logical reasoning to information and draws conclusions",
            llm=llm
        )
        
    def reason(self, query: str, step: str, context: List[Dict[str, Any]]) -> str:
        template = """You are a reasoning agent. Your role is to apply logical analysis to information and draw conclusions.
        
        Given the following step, context, and query, apply logical reasoning to reach a conclusion.
        Show your reasoning process clearly.
        
        Step: {step}
        
        Context:
        {context}
        
        Query: {query}
        
        Reasoning:"""
        
        context_str = "\n\n".join([f"Context {i+1}:\n{item['content']}" for i, item in enumerate(context)])
        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(step=step, query=query, context=context_str)
        response = self.llm.invoke(messages)
        return response.content

class SynthesisAgent(Agent):
    """Agent responsible for combining information and generating final response"""
    def __init__(self, llm):
        super().__init__(
            name="Synthesizer",
            role="Information Synthesizer",
            description="Combines multiple pieces of information into a coherent response",
            llm=llm
        )
        
    def synthesize(self, query: str, reasoning_steps: List[str]) -> str:
        template = """You are a synthesis agent. Your role is to combine multiple pieces of information into a clear, coherent response.
        
        Given the following query and reasoning steps, create a final comprehensive answer.
        The answer should be well-structured and incorporate the key points from each step.
        
        Query: {query}
        
        Reasoning Steps:
        {steps}
        
        Final Answer:"""
        
        steps_str = "\n\n".join([f"Step {i+1}:\n{step}" for i, step in enumerate(reasoning_steps)])
        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(query=query, steps=steps_str)
        response = self.llm.invoke(messages)
        return response.content

def create_agents(llm, vector_store=None):
    """Create and return the set of specialized agents"""
    return {
        "planner": PlannerAgent(llm),
        "researcher": ResearchAgent(llm, vector_store) if vector_store else None,
        "reasoner": ReasoningAgent(llm),
        "synthesizer": SynthesisAgent(llm)
    } 