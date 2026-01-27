# seed_db.py
from app import create_app
from app.database import db, QuestionType, BloomsTaxonomy
from config import Config  # Import your Config class

# Pass the Config class to create_app
app = create_app(Config)

def seed():
    with app.app_context():
        print("Checking/Seeding lookup tables...")
        
        # 1. Add Question Types
        types = ['MCQ', 'FIB', 'Long Answer']
        for t_name in types:
            exists = QuestionType.query.filter_by(type_name=t_name).first()
            if not exists:
                db.session.add(QuestionType(type_name=t_name))
                print(f"Added Type: {t_name}")
        
        # 2. Add Bloom's Levels
        blooms = [
            ('BL-1', 'Remembering'),
            ('BL-2', 'Understanding'),
            ('BL-3', 'Applying'),
            ('BL-4', 'Analyzing'),
            ('BL-5', 'Evaluating'),
            ('BL-6', 'Creating')
        ]
        for code, name in blooms:
            exists = BloomsTaxonomy.query.filter_by(level_code=code).first()
            if not exists:
                db.session.add(BloomsTaxonomy(level_code=code, level_name=name))
                print(f"Added Bloom: {code}")
        
        db.session.commit()
        print("Database seeding completed!")

if __name__ == "__main__":
    seed()