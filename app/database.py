from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

# Association Table for Quizzes and Questions
quiz_question_mapping = db.Table('quiz_question_mapping',
    db.Column('quiz_id', db.Integer, db.ForeignKey('quizzes.id', ondelete='CASCADE'), primary_key=True),
    db.Column('question_id', db.Integer, db.ForeignKey('questions.id', ondelete='CASCADE'), primary_key=True),
    db.Column('points', db.Integer, default=1)
)

class Quiz(db.Model):
    __tablename__ = 'quizzes'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to questions
    questions = db.relationship('Question', secondary=quiz_question_mapping, backref='quizzes')

class Student(db.Model):
    __tablename__ = 'students'
    id = db.Column(db.Integer, primary_key=True)
    roll_no = db.Column(db.String(20), unique=True, nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    class_name = db.Column(db.String(50), nullable=False)
    
    attempts = db.relationship('QuizAttempt', backref='student', cascade="all, delete-orphan")

class QuizAttempt(db.Model):
    __tablename__ = 'quiz_attempts'
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id', ondelete='CASCADE'))
    quiz_id = db.Column(db.Integer, db.ForeignKey('quizzes.id', ondelete='CASCADE'))
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    final_score = db.Column(db.Numeric(5, 2), default=0.0)
    
    responses = db.relationship('StudentResponse', backref='attempt', cascade="all, delete-orphan")

class StudentResponse(db.Model):
    __tablename__ = 'student_responses'
    id = db.Column(db.Integer, primary_key=True)
    attempt_id = db.Column(db.Integer, db.ForeignKey('quiz_attempts.id', ondelete='CASCADE'))
    question_id = db.Column(db.Integer, db.ForeignKey('questions.id'))
    submitted_answer = db.Column(db.Text)
    is_correct = db.Column(db.Boolean, default=False)
    marks_obtained = db.Column(db.Numeric(5, 2), default=0.0)

class BloomsTaxonomy(db.Model):
    __tablename__ = 'blooms_taxonomy'
    id = db.Column(db.Integer, primary_key=True)
    level_code = db.Column(db.String(10), nullable=False)
    level_name = db.Column(db.String(50), nullable=False)

class QuestionType(db.Model):
    __tablename__ = 'question_types'
    id = db.Column(db.Integer, primary_key=True)
    type_name = db.Column(db.String(50), nullable=False)

class Question(db.Model):
    __tablename__ = 'questions'
    id = db.Column(db.Integer, primary_key=True)
    question_text = db.Column(db.Text, nullable=False)
    type_id = db.Column(db.Integer, db.ForeignKey('question_types.id'))
    blooms_id = db.Column(db.Integer, db.ForeignKey('blooms_taxonomy.id'))
    source_document = db.Column(db.String(255))
    
    # Relationships
    mcq_data = db.relationship('McqOption', backref='question', uselist=False)
    text_answer = db.relationship('TextAnswer', backref='question', uselist=False)
    blooms_taxonomy = db.relationship('BloomsTaxonomy', backref='questions')

class McqOption(db.Model):
    __tablename__ = 'mcq_options'
    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('questions.id', ondelete='CASCADE'))
    option_a = db.Column(db.Text, nullable=False)
    option_b = db.Column(db.Text, nullable=False)
    option_c = db.Column(db.Text, nullable=False)
    option_d = db.Column(db.Text, nullable=False)
    correct_option = db.Column(db.String(1))

class TextAnswer(db.Model):
    __tablename__ = 'text_answers'
    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('questions.id', ondelete='CASCADE'))
    answer_content = db.Column(db.Text, nullable=False)
    
class Context(db.Model):
    __tablename__ = 'context'
    
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(255), nullable=False)
    file_content = db.Column(db.Text, nullable=False)  # Stores extracted text content
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return f'<Context {self.file_name}>'
