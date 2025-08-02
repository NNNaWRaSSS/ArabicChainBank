#!/usr/bin/env python3
"""
Data Loader for Qur'an QA System
================================

This module handles loading and processing of:
1. Qur'anic passages from various formats
2. Sahih Bukhari Hadiths
3. Data preprocessing and indexing
"""

import json
import csv
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
from quran_qa_system import Source

class DataLoader:
    """Handles loading and processing of Islamic texts data"""
    
    def __init__(self):
        self.quran_sources = []
        self.hadith_sources = []
    
    def load_quran_from_json(self, file_path: str) -> List[Source]:
        """
        Load Qur'anic passages from JSON format
        
        Expected JSON format:
        {
            "surah": "1",
            "ayah": "1",
            "text": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
            "translation": "In the name of Allah, the Entirely Merciful, the Especially Merciful"
        }
        """
        sources = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                source = Source(
                    id=f"quran_{item['surah']}_{item['ayah']}",
                    text=item['text'],
                    source_type='quran',
                    surah=item['surah'],
                    ayah=item['ayah']
                )
                sources.append(source)
                
        except Exception as e:
            print(f"Error loading Qur'an JSON: {e}")
        
        return sources
    
    def load_quran_from_csv(self, file_path: str) -> List[Source]:
        """
        Load Qur'anic passages from CSV format
        
        Expected CSV columns: surah, ayah, text, translation
        """
        sources = []
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            
            for _, row in df.iterrows():
                source = Source(
                    id=f"quran_{row['surah']}_{row['ayah']}",
                    text=row['text'],
                    source_type='quran',
                    surah=str(row['surah']),
                    ayah=str(row['ayah'])
                )
                sources.append(source)
                
        except Exception as e:
            print(f"Error loading Qur'an CSV: {e}")
        
        return sources
    
    def load_hadith_from_json(self, file_path: str) -> List[Source]:
        """
        Load Sahih Bukhari Hadiths from JSON format
        
        Expected JSON format:
        {
            "hadith_number": "1",
            "book": "كتاب بدء الوحي",
            "chapter": "باب كيف كان بدء الوحي إلى رسول الله صلى الله عليه وسلم",
            "text": "حدثنا الحميدي عبد الله بن الزبير قال حدثنا سفيان قال حدثنا يحيى بن سعيد الأنصاري قال أخبرني محمد بن إبراهيم التيمي أنه سمع علقمة بن وقاص الليثي يقول سمعت عمر بن الخطاب رضي الله عنه على المنبر قال سمعت رسول الله صلى الله عليه وسلم يقول إنما الأعمال بالنيات وإنما لكل امرئ ما نوى فمن كانت هجرته إلى دنيا يصيبها أو إلى امرأة ينكحها فهجرته إلى ما هاجر إليه"
        }
        """
        sources = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                source = Source(
                    id=f"hadith_{item['hadith_number']}",
                    text=item['text'],
                    source_type='hadith',
                    hadith_number=item['hadith_number'],
                    book=item.get('book', ''),
                    chapter=item.get('chapter', '')
                )
                sources.append(source)
                
        except Exception as e:
            print(f"Error loading Hadith JSON: {e}")
        
        return sources
    
    def load_hadith_from_csv(self, file_path: str) -> List[Source]:
        """
        Load Sahih Bukhari Hadiths from CSV format
        
        Expected CSV columns: hadith_number, book, chapter, text
        """
        sources = []
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            
            for _, row in df.iterrows():
                source = Source(
                    id=f"hadith_{row['hadith_number']}",
                    text=row['text'],
                    source_type='hadith',
                    hadith_number=str(row['hadith_number']),
                    book=row.get('book', ''),
                    chapter=row.get('chapter', '')
                )
                sources.append(source)
                
        except Exception as e:
            print(f"Error loading Hadith CSV: {e}")
        
        return sources
    
    def create_sample_data(self) -> tuple:
        """
        Create sample data for testing purposes
        Returns tuple of (quran_sources, hadith_sources)
        """
        # Sample Qur'anic passages
        quran_sources = [
            Source(
                id="quran_1_1",
                text="بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
                source_type='quran',
                surah="1",
                ayah="1"
            ),
            Source(
                id="quran_2_255",
                text="اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ ۚ لَا تَأْخُذُهُ سِنَةٌ وَلَا نَوْمٌ",
                source_type='quran',
                surah="2",
                ayah="255"
            ),
            Source(
                id="quran_112_1",
                text="قُلْ هُوَ اللَّهُ أَحَدٌ",
                source_type='quran',
                surah="112",
                ayah="1"
            ),
            Source(
                id="quran_2_286",
                text="لَا يُكَلِّفُ اللَّهُ نَفْسًا إِلَّا وُسْعَهَا",
                source_type='quran',
                surah="2",
                ayah="286"
            ),
            Source(
                id="quran_49_13",
                text="إِنَّ أَكْرَمَكُمْ عِنْدَ اللَّهِ أَتْقَاكُمْ",
                source_type='quran',
                surah="49",
                ayah="13"
            )
        ]
        
        # Sample Hadiths
        hadith_sources = [
            Source(
                id="hadith_1",
                text="إنما الأعمال بالنيات وإنما لكل امرئ ما نوى",
                source_type='hadith',
                hadith_number="1",
                book="كتاب بدء الوحي",
                chapter="باب كيف كان بدء الوحي"
            ),
            Source(
                id="hadith_2",
                text="بني الإسلام على خمس شهادة أن لا إله إلا الله وأن محمدا رسول الله وإقام الصلاة وإيتاء الزكاة وحج البيت وصوم رمضان",
                source_type='hadith',
                hadith_number="8",
                book="كتاب الإيمان",
                chapter="باب قول النبي صلى الله عليه وسلم بني الإسلام على خمس"
            ),
            Source(
                id="hadith_3",
                text="من حسن إسلام المرء تركه ما لا يعنيه",
                source_type='hadith',
                hadith_number="11",
                book="كتاب الإيمان",
                chapter="باب من حسن إسلام المرء تركه ما لا يعنيه"
            ),
            Source(
                id="hadith_4",
                text="لا يؤمن أحدكم حتى يحب لأخيه ما يحب لنفسه",
                source_type='hadith',
                hadith_number="13",
                book="كتاب الإيمان",
                chapter="باب من الإيمان أن يحب لأخيه ما يحب لنفسه"
            ),
            Source(
                id="hadith_5",
                text="أمرت أن أقاتل الناس حتى يشهدوا أن لا إله إلا الله وأن محمدا رسول الله",
                source_type='hadith',
                hadith_number="25",
                book="كتاب الإيمان",
                chapter="باب قول النبي صلى الله عليه وسلم أمرت أن أقاتل الناس"
            )
        ]
        
        return quran_sources, hadith_sources
    
    def save_sources_to_json(self, sources: List[Source], file_path: str):
        """Save sources to JSON file"""
        data = []
        
        for source in sources:
            item = {
                'id': source.id,
                'text': source.text,
                'source_type': source.source_type
            }
            
            if source.source_type == 'quran':
                item.update({
                    'surah': source.surah,
                    'ayah': source.ayah
                })
            elif source.source_type == 'hadith':
                item.update({
                    'hadith_number': source.hadith_number,
                    'book': source.book,
                    'chapter': source.chapter
                })
            
            data.append(item)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_sources_from_json(self, file_path: str) -> List[Source]:
        """Load sources from JSON file"""
        sources = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                source = Source(
                    id=item['id'],
                    text=item['text'],
                    source_type=item['source_type']
                )
                
                if item['source_type'] == 'quran':
                    source.surah = item.get('surah')
                    source.ayah = item.get('ayah')
                elif item['source_type'] == 'hadith':
                    source.hadith_number = item.get('hadith_number')
                    source.book = item.get('book')
                    source.chapter = item.get('chapter')
                
                sources.append(source)
                
        except Exception as e:
            print(f"Error loading sources from JSON: {e}")
        
        return sources

def main():
    """Test the data loader"""
    loader = DataLoader()
    
    # Create sample data
    quran_sources, hadith_sources = loader.create_sample_data()
    
    print(f"Created {len(quran_sources)} Qur'anic sources")
    print(f"Created {len(hadith_sources)} Hadith sources")
    
    # Save to JSON for testing
    loader.save_sources_to_json(quran_sources, 'sample_quran.json')
    loader.save_sources_to_json(hadith_sources, 'sample_hadith.json')
    
    print("Sample data saved to JSON files")

if __name__ == "__main__":
    main()