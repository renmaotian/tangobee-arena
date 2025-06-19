#!/usr/bin/env python3
"""
TangoBee Local Arena - Local AI Model Evaluation Tool
A peer-to-peer AI evaluation system that you can run locally with your own OpenRouter API key.

Usage:
    python src/local_arena.py --api-key YOUR_OPENROUTER_API_KEY
    or set OPENROUTER_API_KEY environment variable
"""

import requests
import json
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import time
import logging
from pathlib import Path
import random
from typing import List, Dict, Tuple, Optional
import re
import threading
import concurrent.futures
from queue import Queue
import argparse
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/tangobee_local.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TangoBeeLocalArena:
    def __init__(self, api_key: str = None):
        self.base_dir = Path(__file__).parent.parent
        self.api_key = api_key or self.load_api_key()
        self.db_path = self.base_dir / 'data' / 'results.db'
        self.reports_dir = self.base_dir / 'reports'
        self.assets_dir = self.base_dir / 'assets'
        
        # Create directories if they don't exist
        self.db_path.parent.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        self.setup_database()
        
        # OpenRouter API configuration
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Evaluation fields
        self.evaluation_fields = {
            'reasoning': 'logical reasoning and critical thinking',
            'coding': 'programming and software development',
            'language': 'language understanding and communication',
            'mathematics': 'mathematical problem solving',
            'creativity': 'creative thinking and innovation'
        }
        
        # Default models for testing (cheap models)
        self.default_models = [
            "openai/gpt-4o-mini",
            "anthropic/claude-3-5-haiku",
            "deepseek/deepseek-r1"
        ]
        
    def load_api_key(self) -> str:
        """Load OpenRouter API key from environment variable or config file"""
        # Try environment variable first
        api_key = os.getenv('OPENROUTER_API_KEY')
        if api_key:
            return api_key.strip()
            
        # Try config file
        config_file = self.base_dir / 'config' / 'api_key.txt'
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                logger.error(f"Error reading API key from {config_file}: {e}")
                
        raise ValueError(
            "OpenRouter API key not found. Please either:\n"
            "1. Set OPENROUTER_API_KEY environment variable, or\n"
            "2. Create config/api_key.txt with your API key, or\n"
            "3. Pass --api-key argument"
        )
            
    def setup_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Models table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                openrouter_name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Questions table - enhanced schema for distributed evaluation
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                domain TEXT NOT NULL,
                question TEXT NOT NULL,
                evaluation_date TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models (id),
                UNIQUE(model_id, domain, evaluation_date)
            )
        ''')
        
        # Answers table - stores responses to questions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                responder_model_id INTEGER NOT NULL,
                answer TEXT NOT NULL,
                thinking_content TEXT,
                evaluation_date TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (question_id) REFERENCES questions (id),
                FOREIGN KEY (responder_model_id) REFERENCES models (id),
                UNIQUE(question_id, responder_model_id, evaluation_date)
            )
        ''')
        
        # Evaluations table - stores scores given by question creators
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                answer_id INTEGER NOT NULL,
                evaluator_model_id INTEGER NOT NULL,
                score REAL,
                reasoning TEXT,
                evaluation_date TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (answer_id) REFERENCES answers (id),
                FOREIGN KEY (evaluator_model_id) REFERENCES models (id),
                UNIQUE(answer_id, evaluator_model_id, evaluation_date)
            )
        ''')
        
        # Daily scores table - aggregated results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                total_score REAL,
                reasoning_score REAL,
                coding_score REAL,
                language_score REAL,
                mathematics_score REAL,
                creativity_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models (id),
                UNIQUE(model_id, date)
            )
        ''')
        
        # Evaluation status table - track evaluation state
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_date TEXT NOT NULL,
                phase TEXT NOT NULL,
                status TEXT NOT NULL,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database setup completed")

    def clean_model_name_for_display(self, model_name: str) -> str:
        """Format model names with company prefix for display"""
        import re
        
        # Extract company name and model name
        parts = model_name.split('/', 1)
        if len(parts) < 2:
            # No slash, treat as just model name
            return self.format_model_name(model_name)
        
        company = parts[0]
        model = parts[1]
        
        # Clean and format company name
        company_formatted = self.format_company_name(company)
        
        # Clean and format model name
        model_formatted = self.format_model_name(model)
        
        return f"{company_formatted}/{model_formatted}"
    
    def format_company_name(self, company: str) -> str:
        """Format company name for display"""
        company = company.lower().strip()
        
        # Map common company names to their preferred format
        company_map = {
            'anthropic': 'Anthropic',
            'openai': 'OpenAI',
            'google': 'Google',
            'mistralai': 'Mistral AI',
            'meta': 'Meta',
            'cohere': 'Cohere',
            'deepseek': 'DeepSeek'
        }
        
        return company_map.get(company, company.capitalize())
    
    def format_model_name(self, model: str) -> str:
        """Format model name for display"""
        # Clean up common patterns
        model = re.sub(r'[-_]+', ' ', model)  # Replace hyphens and underscores with spaces
        model = re.sub(r'\s+', ' ', model)   # Normalize multiple spaces
        
        # Handle version numbers and special cases
        if 'gpt' in model.lower():
            model = re.sub(r'gpt[\s-]*(\d+)', r'GPT-\1', model, flags=re.IGNORECASE)
        elif 'claude' in model.lower():
            model = re.sub(r'claude[\s-]*(\d+)', r'Claude \1', model, flags=re.IGNORECASE)
        elif 'gemini' in model.lower():
            model = re.sub(r'gemini[\s-]*(\d+)', r'Gemini \1', model, flags=re.IGNORECASE)
        
        # Capitalize first letter of each word
        return ' '.join(word.capitalize() for word in model.split())

    def add_models_to_db(self, models: List[str]):
        """Add models to database if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for model in models:
            display_name = self.clean_model_name_for_display(model)
            cursor.execute(
                'INSERT OR IGNORE INTO models (name, openrouter_name) VALUES (?, ?)',
                (display_name, model)
            )
        
        conn.commit()
        conn.close()
        logger.info(f"Added {len(models)} models to database")
    
    def get_models_from_db(self) -> List[Dict]:
        """Get all models from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, name, openrouter_name FROM models')
        models = [{'id': row[0], 'name': row[1], 'openrouter_name': row[2]} 
                 for row in cursor.fetchall()]
        
        conn.close()
        return models

    def call_openrouter_api(self, model: str, messages: List[Dict], max_tokens: int = 4000, 
                           temperature: float = 0.7, max_retries: int = 3) -> Optional[str]:
        """Call OpenRouter API with retry logic"""
        
        for attempt in range(max_retries):
            try:
                data = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                
                response = requests.post(
                    f"{self.openrouter_base_url}/chat/completions",
                    headers=self.headers,
                    json=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        return result['choices'][0]['message']['content']
                    else:
                        logger.error(f"No choices in response for {model}: {result}")
                        return None
                elif response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited for {model}, waiting {wait_time}s (attempt {attempt + 1})")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"API error for {model}: {response.status_code} - {response.text}")
                    return None
                    
            except Exception as e:
                logger.error(f"Exception calling API for {model} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                
        logger.error(f"Failed to get response from {model} after {max_retries} attempts")
        return None

    def generate_question(self, model: str, domain: str, domain_description: str) -> Optional[str]:
        """Generate a question for a specific domain using a model"""
        
        messages = [
            {
                "role": "system",
                "content": f"""You are helping create questions for evaluating AI models in the domain of {domain_description}.

Generate ONE challenging but fair question that tests {domain_description}. The question should:
1. Be clear and unambiguous
2. Have a definitive correct answer or clear evaluation criteria
3. Be suitable for evaluating AI capabilities
4. Not be too simple or too impossible
5. Be answerable in a reasonable amount of text (not require extremely long responses)

DO NOT include evaluation criteria, scoring rubrics, or hints about how to evaluate answers. Just provide the question itself.

Respond with ONLY the question, nothing else."""
            },
            {
                "role": "user", 
                "content": f"Generate a {domain} question to test {domain_description}."
            }
        ]
        
        response = self.call_openrouter_api(model, messages, max_tokens=500, temperature=0.8)
        if response:
            # Clean up the response to just get the question
            question = response.strip()
            # Remove any prefixes like "Question:" or similar
            question = re.sub(r'^(Question|Q):\s*', '', question, flags=re.IGNORECASE)
            return question
        return None

    def answer_question(self, model: str, question: str, domain: str) -> Tuple[Optional[str], Optional[str]]:
        """Get an answer to a question from a model"""
        
        messages = [
            {
                "role": "system",
                "content": f"""You are answering a question in the domain of {self.evaluation_fields[domain]}. 

Provide a thoughtful, accurate, and complete answer to the question. Show your reasoning and work where appropriate, but keep your response focused and concise.

If the question requires code, provide working code with explanations.
If the question involves calculations, show your work.
If the question requires creative thinking, be original and thoughtful.
If the question involves reasoning, explain your logical steps clearly."""
            },
            {
                "role": "user",
                "content": question
            }
        ]
        
        response = self.call_openrouter_api(model, messages, max_tokens=2000, temperature=0.3)
        return response, None  # No thinking content in regular responses

    def evaluate_answer(self, evaluator_model: str, question: str, answer: str, domain: str) -> Tuple[Optional[float], Optional[str]]:
        """Have a model evaluate an answer to its own question"""
        
        messages = [
            {
                "role": "system",
                "content": f"""You are evaluating an answer to a question you created in the domain of {self.evaluation_fields[domain]}.

Score the answer on a scale of 0-10 where:
- 0-2: Completely wrong, nonsensical, or no understanding shown
- 3-4: Some understanding but major errors or omissions
- 5-6: Partially correct with some errors or missing important details
- 7-8: Mostly correct with minor issues
- 9-10: Excellent, comprehensive, and accurate

Consider:
- Accuracy and correctness
- Completeness of the answer
- Quality of reasoning/explanation
- Appropriateness for the domain

Respond with ONLY a number from 0-10 (can include decimals like 7.5), followed by a brief explanation of your reasoning in 1-2 sentences."""
            },
            {
                "role": "user",
                "content": f"""Question: {question}

Answer to evaluate: {answer}

Provide your score (0-10) and brief reasoning."""
            }
        ]
        
        response = self.call_openrouter_api(evaluator_model, messages, max_tokens=200, temperature=0.2)
        if response:
            try:
                # Extract score and reasoning
                lines = response.strip().split('\n')
                first_line = lines[0].strip()
                
                # Try to extract score from first line
                score_match = re.search(r'(\d+(?:\.\d+)?)', first_line)
                if score_match:
                    score = float(score_match.group(1))
                    # Ensure score is in valid range
                    score = max(0, min(10, score))
                    
                    # Get reasoning (rest of the response)
                    reasoning = '\n'.join(lines[1:]).strip() if len(lines) > 1 else first_line
                    return score, reasoning
                else:
                    logger.error(f"Could not extract score from response: {response}")
                    return None, None
                    
            except Exception as e:
                logger.error(f"Error parsing evaluation response: {e}")
                return None, None
        
        return None, None

    def run_evaluation_phase_1_questions(self, models: List[Dict], evaluation_date: str):
        """Phase 1: Generate questions from all models for all domains"""
        logger.info("Phase 1: Generating questions")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing questions for this date
        cursor.execute('DELETE FROM questions WHERE evaluation_date = ?', (evaluation_date,))
        
        total_questions = len(models) * len(self.evaluation_fields)
        generated_questions = 0
        
        for model in models:
            model_id = model['id']
            model_name = model['openrouter_name']
            
            logger.info(f"Generating questions for {model['name']}")
            
            for domain, description in self.evaluation_fields.items():
                logger.info(f"  Generating {domain} question...")
                
                question = self.generate_question(model_name, domain, description)
                
                if question:
                    cursor.execute('''
                        INSERT INTO questions (model_id, domain, question, evaluation_date)
                        VALUES (?, ?, ?, ?)
                    ''', (model_id, domain, question, evaluation_date))
                    generated_questions += 1
                    logger.info(f"    Generated: {question[:100]}...")
                else:
                    logger.error(f"    Failed to generate {domain} question for {model['name']}")
                
                # Small delay to avoid rate limiting
                time.sleep(1)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Phase 1 complete: Generated {generated_questions}/{total_questions} questions")
        return generated_questions > 0

    def run_evaluation_phase_2_answers(self, models: List[Dict], evaluation_date: str):
        """Phase 2: Collect answers from all models to all questions"""
        logger.info("Phase 2: Collecting answers")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all questions for this evaluation date
        cursor.execute('''
            SELECT q.id, q.domain, q.question, m.name as questioner_name
            FROM questions q
            JOIN models m ON q.model_id = m.id
            WHERE q.evaluation_date = ?
        ''', (evaluation_date,))
        questions = cursor.fetchall()
        
        # Clear existing answers for this date
        cursor.execute('DELETE FROM answers WHERE evaluation_date = ?', (evaluation_date,))
        
        total_answers_needed = len(questions) * len(models)
        collected_answers = 0
        
        for question_id, domain, question_text, questioner_name in questions:
            logger.info(f"Getting answers for {domain} question from {questioner_name}: {question_text[:100]}...")
            
            for model in models:
                responder_id = model['id']
                responder_name = model['openrouter_name']
                
                logger.info(f"  Getting answer from {model['name']}")
                
                answer, thinking = self.answer_question(responder_name, question_text, domain)
                
                if answer:
                    cursor.execute('''
                        INSERT INTO answers (question_id, responder_model_id, answer, thinking_content, evaluation_date)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (question_id, responder_id, answer, thinking, evaluation_date))
                    collected_answers += 1
                    logger.info(f"    Collected answer: {answer[:100]}...")
                else:
                    logger.error(f"    Failed to get answer from {model['name']}")
                
                # Small delay to avoid rate limiting
                time.sleep(1)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Phase 2 complete: Collected {collected_answers}/{total_answers_needed} answers")
        return collected_answers > 0

    def run_evaluation_phase_3_scoring(self, models: List[Dict], evaluation_date: str):
        """Phase 3: Have each model evaluate answers to their own questions"""
        logger.info("Phase 3: Evaluating answers")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing evaluations for this date
        cursor.execute('DELETE FROM evaluations WHERE evaluation_date = ?', (evaluation_date,))
        
        total_evaluations = 0
        completed_evaluations = 0
        
        for model in models:
            evaluator_id = model['id']
            evaluator_name = model['openrouter_name']
            
            # Get questions created by this model
            cursor.execute('''
                SELECT q.id, q.domain, q.question
                FROM questions q
                WHERE q.model_id = ? AND q.evaluation_date = ?
            ''', (evaluator_id, evaluation_date))
            questions = cursor.fetchall()
            
            logger.info(f"Model {model['name']} evaluating answers to {len(questions)} of its questions")
            
            for question_id, domain, question_text in questions:
                # Get all answers to this question (excluding the evaluator's own answer)
                cursor.execute('''
                    SELECT a.id, a.answer, m.name as responder_name
                    FROM answers a
                    JOIN models m ON a.responder_model_id = m.id
                    WHERE a.question_id = ? AND a.responder_model_id != ? AND a.evaluation_date = ?
                ''', (question_id, evaluator_id, evaluation_date))
                answers = cursor.fetchall()
                
                total_evaluations += len(answers)
                
                logger.info(f"  Evaluating {len(answers)} answers to {domain} question: {question_text[:100]}...")
                
                for answer_id, answer_text, responder_name in answers:
                    logger.info(f"    Evaluating answer from {responder_name}")
                    
                    score, reasoning = self.evaluate_answer(evaluator_name, question_text, answer_text, domain)
                    
                    if score is not None:
                        cursor.execute('''
                            INSERT INTO evaluations (answer_id, evaluator_model_id, score, reasoning, evaluation_date)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (answer_id, evaluator_id, score, reasoning, evaluation_date))
                        completed_evaluations += 1
                        logger.info(f"      Score: {score}/10 - {reasoning[:100]}...")
                    else:
                        logger.error(f"      Failed to get evaluation")
                    
                    # Small delay to avoid rate limiting
                    time.sleep(1)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Phase 3 complete: Completed {completed_evaluations}/{total_evaluations} evaluations")
        return completed_evaluations > 0

    def calculate_daily_scores(self, evaluation_date: str):
        """Calculate and store daily aggregated scores"""
        logger.info("Calculating daily scores")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing daily scores for this date
        cursor.execute('DELETE FROM daily_scores WHERE date = ?', (evaluation_date,))
        
        # Calculate scores for each model
        for model in self.get_models_from_db():
            model_id = model['id']
            
            # Get all evaluations where this model was the responder
            query = '''
                SELECT 
                    q.domain,
                    e.score
                FROM evaluations e
                JOIN answers a ON e.answer_id = a.id
                JOIN questions q ON a.question_id = q.id
                WHERE a.responder_model_id = ? AND e.evaluation_date = ?
            '''
            
            cursor.execute(query, (model_id, evaluation_date))
            evaluations = cursor.fetchall()
            
            if not evaluations:
                logger.warning(f"No evaluations found for {model['name']}")
                continue
            
            # Calculate domain averages
            domain_scores = {}
            for domain in self.evaluation_fields.keys():
                domain_evaluations = [score for d, score in evaluations if d == domain and score is not None]
                if domain_evaluations:
                    domain_scores[domain] = sum(domain_evaluations) / len(domain_evaluations)
                else:
                    domain_scores[domain] = None
            
            # Calculate total average (only from domains with scores)
            valid_scores = [score for score in domain_scores.values() if score is not None]
            total_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
            
            # Insert daily score
            cursor.execute('''
                INSERT INTO daily_scores 
                (model_id, date, total_score, reasoning_score, coding_score, language_score, mathematics_score, creativity_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_id, evaluation_date, total_score,
                domain_scores.get('reasoning'), domain_scores.get('coding'), 
                domain_scores.get('language'), domain_scores.get('mathematics'), 
                domain_scores.get('creativity')
            ))
            
            total_score_str = f"{total_score:.2f}" if total_score else "N/A"
            logger.info(f"Calculated scores for {model['name']}: Total={total_score_str}")
        
        conn.commit()
        conn.close()
        logger.info("Daily scores calculation complete")

    def generate_visualizations(self, evaluation_date: str):
        """Generate heatmaps and trend plots"""
        logger.info("Generating visualizations")
        
        viz_dir = self.reports_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        
        # Generate heatmaps
        self.generate_heatmaps(conn, viz_dir, evaluation_date)
        
        # Generate trend plots
        self.generate_trend_plots(conn, viz_dir)
        
        conn.close()
        logger.info("Visualizations generated")

    def generate_heatmaps(self, conn, viz_dir: Path, evaluation_date: str):
        """Generate performance heatmaps"""
        
        # Get evaluation data for heatmaps
        query = '''
            SELECT 
                evaluator.name as evaluator_name,
                responder.name as responder_name,
                q.domain,
                e.score
            FROM evaluations e
            JOIN answers a ON e.answer_id = a.id
            JOIN questions q ON a.question_id = q.id
            JOIN models evaluator ON e.evaluator_model_id = evaluator.id
            JOIN models responder ON a.responder_model_id = responder.id
            WHERE e.evaluation_date = ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(evaluation_date,))
        
        if df.empty:
            logger.warning("No evaluation data found for heatmaps")
            return
        
        # Create heatmaps for each domain and overall
        domains = list(self.evaluation_fields.keys()) + ['overall']
        
        for domain in domains:
            plt.figure(figsize=(12, 10))
            
            if domain == 'overall':
                # Overall heatmap: average across all domains
                heatmap_data = df.groupby(['evaluator_name', 'responder_name'])['score'].mean().unstack(fill_value=0)
                title = 'Overall Performance Heatmap'
            else:
                # Domain-specific heatmap
                domain_df = df[df['domain'] == domain]
                if domain_df.empty:
                    continue
                heatmap_data = domain_df.groupby(['evaluator_name', 'responder_name'])['score'].mean().unstack(fill_value=0)
                title = f'{domain.capitalize()} Performance Heatmap'
            
            # Create heatmap
            sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=5, 
                       vmin=0, vmax=10, fmt='.1f', cbar_kws={'label': 'Score'})
            
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('Model Being Evaluated', fontweight='bold')
            plt.ylabel('Evaluating Model', fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save heatmap
            plt.savefig(viz_dir / f'heatmap_{domain}_{evaluation_date}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()

    def generate_trend_plots(self, conn, viz_dir: Path):
        """Generate trend line plots"""
        
        # Get data for last 6 months (180 days)
        query = '''
            SELECT 
                ds.date,
                m.name as model_name,
                ds.total_score,
                ds.reasoning_score,
                ds.coding_score,
                ds.language_score,
                ds.mathematics_score,
                ds.creativity_score
            FROM daily_scores ds
            JOIN models m ON ds.model_id = m.id
            WHERE ds.date >= date('now', '-180 days')
            ORDER BY ds.date, m.name
        '''
        
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            logger.warning("No historical data found for trend plots")
            return
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Total scores trend
        plt.figure(figsize=(14, 8))
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]
            plt.plot(model_data['date'], model_data['total_score'], 
                    marker='o', label=model, linewidth=2)
        
        plt.title('Total Score Trends (Last 6 Months)', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontweight='bold')
        plt.ylabel('Total Score', fontweight='bold')
        plt.ylim(0, 10)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(viz_dir / 'trends_total.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Domain-specific trends
        domains = ['reasoning_score', 'coding_score', 'language_score', 'mathematics_score', 'creativity_score']
        domain_names = ['Reasoning', 'Coding', 'Language', 'Mathematics', 'Creativity']
        
        for domain, domain_name in zip(domains, domain_names):
            plt.figure(figsize=(14, 8))
            for model in df['model_name'].unique():
                model_data = df[df['model_name'] == model]
                plt.plot(model_data['date'], model_data[domain], 
                        marker='o', label=model, linewidth=2)
            
            plt.title(f'{domain_name} Score Trends (Last 6 Months)', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontweight='bold')
            plt.ylabel(f'{domain_name} Score', fontweight='bold')
            plt.ylim(0, 10)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(viz_dir / f'trends_{domain.replace("_score", "")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()

    def generate_html_report(self, evaluation_date: str):
        """Generate HTML report with results and visualizations"""
        logger.info("Generating HTML report")
        
        conn = sqlite3.connect(self.db_path)
        
        # Get current results
        query = '''
            SELECT 
                m.name as model_name,
                ds.total_score,
                ds.reasoning_score,
                ds.coding_score,
                ds.language_score,
                ds.mathematics_score,
                ds.creativity_score
            FROM daily_scores ds
            JOIN models m ON ds.model_id = m.id
            WHERE ds.date = ?
            ORDER BY ds.total_score DESC NULLS LAST
        '''
        
        results_df = pd.read_sql_query(query, conn, params=(evaluation_date,))
        
        # Get evaluation statistics
        stats_query = '''
            SELECT 
                COUNT(DISTINCT q.id) as total_questions,
                COUNT(DISTINCT a.id) as total_answers,
                COUNT(DISTINCT e.id) as total_evaluations
            FROM questions q
            LEFT JOIN answers a ON q.id = a.question_id AND a.evaluation_date = ?
            LEFT JOIN evaluations e ON a.id = e.answer_id AND e.evaluation_date = ?
            WHERE q.evaluation_date = ?
        '''
        
        cursor = conn.cursor()
        cursor.execute(stats_query, (evaluation_date, evaluation_date, evaluation_date))
        stats = cursor.fetchone()
        
        conn.close()
        
        # Load TangoBee logo as base64
        logo_path = self.assets_dir / 'logo.png'
        logo_base64 = ""
        if logo_path.exists():
            with open(logo_path, 'rb') as f:
                logo_base64 = base64.b64encode(f.read()).decode()
        
        # Generate HTML
        html_content = self.create_html_report_content(
            results_df, evaluation_date, stats, logo_base64
        )
        
        # Save HTML report
        report_path = self.reports_dir / f'tangobee_report_{evaluation_date}.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {report_path}")
        return report_path

    def create_html_report_content(self, results_df, evaluation_date, stats, logo_base64):
        """Create the HTML content for the report"""
        
        # Create results table
        results_table = ""
        if not results_df.empty:
            results_table = "<tbody>"
            for i, row in results_df.iterrows():
                rank = i + 1
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""
                
                # Format scores
                total_score = f"{row['total_score']:.2f}" if pd.notna(row['total_score']) else 'N/A'
                reasoning_score = f"{row['reasoning_score']:.2f}" if pd.notna(row['reasoning_score']) else 'N/A'
                coding_score = f"{row['coding_score']:.2f}" if pd.notna(row['coding_score']) else 'N/A'
                language_score = f"{row['language_score']:.2f}" if pd.notna(row['language_score']) else 'N/A'
                mathematics_score = f"{row['mathematics_score']:.2f}" if pd.notna(row['mathematics_score']) else 'N/A'
                creativity_score = f"{row['creativity_score']:.2f}" if pd.notna(row['creativity_score']) else 'N/A'
                
                results_table += f"""
                <tr>
                    <td class="rank">{medal} #{rank}</td>
                    <td class="model-name">{row['model_name']}</td>
                    <td class="score total-score">{total_score}</td>
                    <td class="score">{reasoning_score}</td>
                    <td class="score">{coding_score}</td>
                    <td class="score">{language_score}</td>
                    <td class="score">{mathematics_score}</td>
                    <td class="score">{creativity_score}</td>
                </tr>
                """
            results_table += "</tbody>"
        else:
            results_table = "<tbody><tr><td colspan='8'>No results found</td></tr></tbody>"
        
        # Create visualization gallery
        viz_dir = self.reports_dir / 'visualizations'
        viz_gallery = ""
        
        # Add heatmaps
        domains = ['overall', 'reasoning', 'coding', 'language', 'mathematics', 'creativity']
        for domain in domains:
            heatmap_file = viz_dir / f'heatmap_{domain}_{evaluation_date}.png'
            if heatmap_file.exists():
                with open(heatmap_file, 'rb') as f:
                    img_base64 = base64.b64encode(f.read()).decode()
                viz_gallery += f"""
                <div class="visualization">
                    <h3>{domain.capitalize()} Performance Heatmap</h3>
                    <img src="data:image/png;base64,{img_base64}" alt="{domain} heatmap" />
                </div>
                """
        
        # Add trend plots
        trend_files = ['trends_total.png', 'trends_reasoning.png', 'trends_coding.png', 
                      'trends_language.png', 'trends_mathematics.png', 'trends_creativity.png']
        trend_names = ['Total Score', 'Reasoning', 'Coding', 'Language', 'Mathematics', 'Creativity']
        
        for trend_file, trend_name in zip(trend_files, trend_names):
            trend_path = viz_dir / trend_file
            if trend_path.exists():
                with open(trend_path, 'rb') as f:
                    img_base64 = base64.b64encode(f.read()).decode()
                viz_gallery += f"""
                <div class="visualization">
                    <h3>{trend_name} Trends</h3>
                    <img src="data:image/png;base64,{img_base64}" alt="{trend_name} trends" />
                </div>
                """
        
        # HTML template
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TangoBee Local Arena Report - {evaluation_date}</title>
    <style>
        :root {{
            --primary-gold: #FFD700;
            --secondary-gold: #FFC107;
            --accent-gold: #FF8F00;
            --dark-gold: #B8860B;
            --background: #FAFAFA;
            --white: #FFFFFF;
            --text-dark: #2C3E50;
            --text-light: #7F8C8D;
            --border: #E0E0E0;
            --shadow: rgba(0, 0, 0, 0.1);
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, var(--background) 0%, #FFF8DC 100%);
            color: var(--text-dark);
            line-height: 1.6;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        .header {{
            text-align: center;
            margin-bottom: 40px;
            background: var(--white);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 8px 32px var(--shadow);
        }}

        .logo {{
            width: 100px;
            height: 100px;
            margin: 0 auto 20px;
            display: block;
        }}

        .title {{
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(135deg, var(--primary-gold), var(--accent-gold));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }}

        .subtitle {{
            font-size: 1.2rem;
            color: var(--text-light);
            margin-bottom: 20px;
        }}

        .evaluation-info {{
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
            margin-top: 20px;
        }}

        .info-item {{
            text-align: center;
        }}

        .info-label {{
            font-size: 0.9rem;
            color: var(--text-light);
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .info-value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: var(--accent-gold);
        }}

        .section {{
            background: var(--white);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px var(--shadow);
        }}

        .section-title {{
            font-size: 1.8rem;
            font-weight: bold;
            color: var(--text-dark);
            margin-bottom: 20px;
            text-align: center;
        }}

        .results-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
        }}

        .results-table th {{
            background: linear-gradient(135deg, var(--primary-gold), var(--secondary-gold));
            color: var(--text-dark);
            font-weight: bold;
            padding: 15px 12px;
            text-align: center;
            border: none;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .results-table th:first-child {{
            border-radius: 10px 0 0 0;
        }}

        .results-table th:last-child {{
            border-radius: 0 10px 0 0;
        }}

        .results-table td {{
            padding: 15px 12px;
            text-align: center;
            border-bottom: 1px solid var(--border);
        }}

        .results-table tbody tr:hover {{
            background: linear-gradient(135deg, #FFF8DC 0%, #FAFAFA 100%);
        }}

        .rank {{
            font-weight: bold;
            font-size: 1.1rem;
        }}

        .model-name {{
            font-weight: bold;
            text-align: left !important;
            color: var(--text-dark);
        }}

        .score {{
            font-weight: bold;
            color: var(--accent-gold);
        }}

        .total-score {{
            background: linear-gradient(135deg, var(--primary-gold), var(--secondary-gold));
            color: var(--text-dark) !important;
            border-radius: 8px;
            font-size: 1.1rem;
        }}

        .visualizations {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-top: 20px;
        }}

        .visualization {{
            text-align: center;
        }}

        .visualization h3 {{
            margin-bottom: 15px;
            font-size: 1.3rem;
            color: var(--text-dark);
        }}

        .visualization img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 16px var(--shadow);
        }}

        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: var(--text-light);
            font-size: 0.9rem;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .title {{
                font-size: 2rem;
            }}
            
            .evaluation-info {{
                gap: 15px;
            }}
            
            .results-table {{
                font-size: 0.8rem;
            }}
            
            .results-table th,
            .results-table td {{
                padding: 8px 6px;
            }}
            
            .visualizations {{
                grid-template-columns: 1fr;
                gap: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <img src="data:image/png;base64,{logo_base64}" alt="TangoBee Logo" class="logo" />
            <h1 class="title">TangoBee Local Arena</h1>
            <p class="subtitle">Peer-to-Peer AI Model Evaluation Report</p>
            
            <div class="evaluation-info">
                <div class="info-item">
                    <div class="info-label">Evaluation Date</div>
                    <div class="info-value">{evaluation_date}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Questions Generated</div>
                    <div class="info-value">{stats[0] if stats else 0}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Answers Collected</div>
                    <div class="info-value">{stats[1] if stats else 0}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Evaluations Made</div>
                    <div class="info-value">{stats[2] if stats else 0}</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">🏆 Leaderboard Results</h2>
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Total Score</th>
                        <th>Reasoning</th>
                        <th>Coding</th>
                        <th>Language</th>
                        <th>Mathematics</th>
                        <th>Creativity</th>
                    </tr>
                </thead>
                {results_table}
            </table>
        </div>

        <div class="section">
            <h2 class="section-title">📊 Performance Analysis</h2>
            <div class="visualizations">
                {viz_gallery}
            </div>
        </div>

        <div class="footer">
            <p>Generated by TangoBee Local Arena • Peer-to-Peer AI Evaluation</p>
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_content

    def run_full_evaluation(self, models: List[str] = None):
        """Run complete evaluation cycle"""
        if models is None:
            models = self.default_models
            
        evaluation_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Starting TangoBee Local Arena evaluation for {evaluation_date}")
        logger.info(f"Models: {models}")
        
        # Check if results already exist for today
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM daily_scores WHERE date = ?', (evaluation_date,))
        existing_results = cursor.fetchone()[0]
        conn.close()
        
        if existing_results > 0:
            logger.info(f"Results already exist for {evaluation_date}. Overwriting...")
            # Clear existing results for today
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM daily_scores WHERE date = ?', (evaluation_date,))
            cursor.execute('DELETE FROM evaluations WHERE evaluation_date = ?', (evaluation_date,))
            cursor.execute('DELETE FROM answers WHERE evaluation_date = ?', (evaluation_date,))
            cursor.execute('DELETE FROM questions WHERE evaluation_date = ?', (evaluation_date,))
            conn.commit()
            conn.close()
        
        try:
            # Add models to database
            self.add_models_to_db(models)
            db_models = self.get_models_from_db()
            
            # Filter to only requested models
            active_models = [m for m in db_models if m['openrouter_name'] in models]
            
            if len(active_models) < 2:
                raise ValueError("Need at least 2 models for evaluation")
            
            logger.info(f"Running evaluation with {len(active_models)} models")
            
            # Phase 1: Generate questions
            if not self.run_evaluation_phase_1_questions(active_models, evaluation_date):
                raise RuntimeError("Phase 1 failed: Could not generate questions")
            
            # Phase 2: Collect answers
            if not self.run_evaluation_phase_2_answers(active_models, evaluation_date):
                raise RuntimeError("Phase 2 failed: Could not collect answers")
            
            # Phase 3: Score answers
            if not self.run_evaluation_phase_3_scoring(active_models, evaluation_date):
                raise RuntimeError("Phase 3 failed: Could not score answers")
            
            # Calculate daily scores
            self.calculate_daily_scores(evaluation_date)
            
            # Generate visualizations
            self.generate_visualizations(evaluation_date)
            
            # Generate HTML report
            report_path = self.generate_html_report(evaluation_date)
            
            logger.info("="*60)
            logger.info("🎉 TangoBee Local Arena evaluation completed successfully! 🎉")
            logger.info(f"📊 Report saved to: {report_path}")
            logger.info("="*60)
            
            return report_path
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='TangoBee Local Arena - AI Model Evaluation')
    parser.add_argument('--api-key', help='OpenRouter API key')
    parser.add_argument('--models', nargs='+', help='Models to evaluate', 
                       default=["openai/gpt-4o-mini", "anthropic/claude-3-5-haiku", "deepseek/deepseek-r1"])
    parser.add_argument('--config', help='Path to config file')
    
    args = parser.parse_args()
    
    try:
        arena = TangoBeeLocalArena(api_key=args.api_key)
        report_path = arena.run_full_evaluation(args.models)
        
        print(f"\n🏆 Evaluation complete! Open the report: {report_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()