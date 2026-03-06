"""
PDF CHATBOT TEST CASES VE EDGE CASES
Profesyonel test framework - 100 dokuman için ölçeklenebilir
"""

from typing import List, Dict, Tuple
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json

# ===== TEST KATEGORİLERİ =====
class TestCategory(Enum):
    """Test kategorileri"""
    BASIC_RETRIEVAL = "basic_retrieval"          # Basit bilgi alma
    COMPLEX_REASONING = "complex_reasoning"      # Karmaşık mantık
    HALLUCINATION_GUARD = "hallucination_guard"  # Sahte cevapları engelleme
    EDGE_CASE = "edge_case"                      # Sınır durumlar
    PERFORMANCE = "performance"                  # Performans
    ROBUSTNESS = "robustness"                    # Sağlamlık

# ===== TEST SONUÇ VE METRİKLER =====
@dataclass
class TestResult:
    """Bir test'in sonucu"""
    test_name: str
    test_id: str
    category: TestCategory
    question: str
    answer: str
    expected_pattern: str
    passed: bool
    similarity_score: float
    retrieval_time: float
    llm_time: float
    retrieved_chunks: int
    confidence_score: float
    error_message: str = ""

# ===== TEST CASE TANARI =====
class TestValidator(ABC):
    """Test sonucu doğrulayıcı (abstract)"""
    
    @abstractmethod
    def validate(self, result: TestResult) -> bool:
        """Sonucu doğrula"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Validator adı"""
        pass

# ===== SPESIFIK VALİDATÖRLER =====
class ResponseNotEmptyValidator(TestValidator):
    """Cevap boş olmamalı"""
    def validate(self, result: TestResult) -> bool:
        return len(result.answer.strip()) > 10
    
    def get_name(self) -> str:
        return "Cevap boş değil"

class NoHallucinationValidator(TestValidator):
    """Dokümanda olmayan bilgi söylememeli"""
    def validate(self, result: TestResult) -> bool:
        hallucination_patterns = [
            "i'm not sure",
            "i don't know",
            "not mentioned",
            "not found",
            "i cannot",
            "belirtilmedi",
            "bulunmadı",
            "bilemiyorum",
            "emin değilim",
            "yer almıyor"
        ]
        answer_lower = result.answer.lower()
        
        # Belgede olmayan bilgiyi söylemediyse ve cevabı bok ayrıca ne veya "bilemiyorum" derse geçer
        contains_denial = any(p in answer_lower for p in hallucination_patterns)
        has_similarity = result.similarity_score > 0.5
        
        # Yüksek benzerlik + detaylı cevap VEYA düşük benzerlik + çekince cevap
        return (has_similarity and len(result.answer) > 20) or (not has_similarity and contains_denial)

class SimilarityThresholdValidator(TestValidator):
    """Benzerlik skoru minimum threshold'ü geçmeli"""
    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold
    
    def validate(self, result: TestResult) -> bool:
        return result.similarity_score >= self.threshold
    
    def get_name(self) -> str:
        return f"Benzerlik >= {self.threshold}"

class SourceFoundValidator(TestValidator):
    """En az 1 kaynak bulunmalı"""
    def validate(self, result: TestResult) -> bool:
        return result.retrieved_chunks > 0
    
    def get_name(self) -> str:
        return "Kaynak bulundu"

class PerformanceValidator(TestValidator):
    """Çalışma süresi kabul edilebilir sınırda"""
    def __init__(self, max_time: float = 30.0):
        self.max_time = max_time
    
    def validate(self, result: TestResult) -> bool:
        return (result.retrieval_time + result.llm_time) < self.max_time
    
    def get_name(self) -> str:
        return f"Hız < {self.max_time}s"

class RelevanceValidator(TestValidator):
    """Cevap soruyla ilgili (manuel check gerekir)"""
    def validate(self, result: TestResult) -> bool:
        # Otomatik: cevap uzunluğu ve benzerlik
        return result.similarity_score > 0.3 and len(result.answer) > 15
    
    def get_name(self) -> str:
        return "İlgili cevap"

# ===== TEST CASE SETLERİ =====
class ComprehensiveTestSuite:
    """Kapsamlı test paketi"""
    
    @staticmethod
    def get_basic_tests() -> List[Dict]:
        """Basit test case'leri (her dokuman için)"""
        return [
            {
                'id': 'basic_001',
                'name': 'Özet Soru',
                'question': 'Bu belge hakkında kısaca bilgi ver',
                'category': TestCategory.BASIC_RETRIEVAL,
                'validators': ['ResponseNotEmpty', 'SourceFound'],
                'expected_pattern': 'summary',
                'difficulty': 'easy',
                'description': 'Dokümandaki ana konuyu bulması gerekir'
            },
            {
                'id': 'basic_002',
                'name': 'Anahtar Noktalar',
                'question': 'Belgede ana noktalar nelerdir?',
                'category': TestCategory.BASIC_RETRIEVAL,
                'validators': ['ResponseNotEmpty', 'Relevance'],
                'expected_pattern': 'key_points',
                'difficulty': 'easy',
                'description': 'Önemli noktaları listelemeli'
            },
            {
                'id': 'basic_003',
                'name': 'Tanım Arama',
                'question': 'Belgede tanımlanan terimler nelerdir?',
                'category': TestCategory.BASIC_RETRIEVAL,
                'validators': ['ResponseNotEmpty', 'Relevance'],
                'expected_pattern': 'definitions',
                'difficulty': 'easy',
                'description': 'Tanımlanan terimleri bulmalı'
            },
        ]
    
    @staticmethod
    def get_complex_tests() -> List[Dict]:
        """Karmaşık test case'leri"""
        return [
            {
                'id': 'complex_001',
                'name': 'İlişki Analizi',
                'question': 'Belgede anlatılan konseptler arasında nasıl ilişkiler var?',
                'category': TestCategory.COMPLEX_REASONING,
                'validators': ['ResponseNotEmpty', 'Relevance', 'Similarity'],
                'expected_pattern': 'relationships',
                'difficulty': 'hard',
                'description': 'Konseptler arasındaki ilişkileri anlayıp açıklamalı'
            },
            {
                'id': 'complex_002',
                'name': 'Çıkarım',
                'question': 'Belgede ima edilen ne var, açık olmayan?',
                'category': TestCategory.COMPLEX_REASONING,
                'validators': ['ResponseNotEmpty', 'NoHallucination'],
                'expected_pattern': 'inference',
                'difficulty': 'hard',
                'description': 'Mantıklı çıkarımlar yapmalı ama uydurmamalı'
            },
            {
                'id': 'complex_003',
                'name': 'Karşılaştırma',
                'question': 'Belgede farklı bakış açıları ya da pozisyonlar karşılaştırılıyor mu?',
                'category': TestCategory.COMPLEX_REASONING,
                'validators': ['ResponseNotEmpty', 'Relevance'],
                'expected_pattern': 'comparison',
                'difficulty': 'hard',
                'description': 'Karşılaştırmaları bulup açıklamalı'
            },
        ]
    
    @staticmethod
    def get_hallucination_tests() -> List[Dict]:
        """Hallucination (sahte cevap) testleri"""
        return [
            {
                'id': 'halluc_001',
                'name': 'Olmayan Bilgi - Güvenli Reddetme',
                'question': 'Belgede hiç anlatılmayan bir konu ne? (Varsayımsal soru)',
                'category': TestCategory.HALLUCINATION_GUARD,
                'validators': ['NoHallucination'],
                'expected_pattern': 'denial_or_low_confidence',
                'difficulty': 'hard',
                'description': 'Olmayan bilgiyi uydurmak yerine reddetmeli'
            },
            {
                'id': 'halluc_002',
                'name': 'Kontrol - Gerçek Bilgi',
                'question': 'Belgede gerçekten anlatılan bir konu nedir?',
                'category': TestCategory.HALLUCINATION_GUARD,
                'validators': ['ResponseNotEmpty', 'SourceFound'],
                'expected_pattern': 'accurate_answer',
                'difficulty': 'medium',
                'description': 'Gerçek bilgiyi doğru şekilde sunmalı'
            },
            {
                'id': 'halluc_003',
                'name': 'Kısmi Bilgi',
                'question': 'Belgede kısmen anlatılan ancak tam olmayan bir konuyu açıkla',
                'category': TestCategory.HALLUCINATION_GUARD,
                'validators': ['ResponseNotEmpty', 'Relevance', 'NoHallucination'],
                'expected_pattern': 'partial_answer_with_caveats',
                'difficulty': 'hard',
                'description': 'Parçalı bilgiyi kısıtlı bir şekilde sunmalı'
            },
        ]
    
    @staticmethod
    def get_edge_case_tests() -> List[Dict]:
        """Sınır durumlar (edge cases)"""
        return [
            {
                'id': 'edge_001',
                'name': 'Çok Kısa Soru',
                'question': 'Ne?',
                'category': TestCategory.EDGE_CASE,
                'validators': ['ResponseNotEmpty'],
                'expected_pattern': 'tolerance',
                'difficulty': 'medium',
                'description': 'Kısa soru bile cevaplandırabilmeli'
            },
            {
                'id': 'edge_002',
                'name': 'Çok Uzun Soru',
                'question': 'Bu belgede anlatılan konuların en önemlisi hangisidir ve neden önemlidir? Lütfen detaylı bir şekilde açıklayın ve diğer konularla karşılaştırın.',
                'category': TestCategory.EDGE_CASE,
                'validators': ['ResponseNotEmpty', 'Relevance'],
                'expected_pattern': 'complex_answer',
                'difficulty': 'hard',
                'description': 'Karmaşık sorulara karmaşık cevaplar verebilmeli'
            },
            {
                'id': 'edge_003',
                'name': 'Sayısal Sorgu',
                'question': 'Belgede hangi sayılar ve istatistikler var?',
                'category': TestCategory.EDGE_CASE,
                'validators': ['ResponseNotEmpty', 'SourceFound'],
                'expected_pattern': 'numerical',
                'difficulty': 'medium',
                'description': 'Sayıları doğru bir şekilde çıkartmalı'
            },
            {
                'id': 'edge_004',
                'name': 'Yazım Hataları ile Soru',
                'question': 'Belgeede ana konu nedir?',  # "Belgeede" yazım hatası
                'category': TestCategory.EDGE_CASE,
                'validators': ['ResponseNotEmpty'],
                'expected_pattern': 'robustness',
                'difficulty': 'medium',
                'description': 'Yazım hatalarına dayanıklı olmalı'
            },
            {
                'id': 'edge_005',
                'name': 'Boş Soru',
                'question': '',
                'category': TestCategory.EDGE_CASE,
                'validators': [],  # Özel handling
                'expected_pattern': 'graceful_fail',
                'difficulty': 'easy',
                'description': 'Boş soruya nazikçe hata vermeli'
            },
            {
                'id': 'edge_006',
                'name': 'Çelişkili Soru',
                'question': 'Belgede X hem doğru hem yanlış olarak anlatılıyor mu?',
                'category': TestCategory.EDGE_CASE,
                'validators': ['ResponseNotEmpty'],
                'expected_pattern': 'contradiction_handling',
                'difficulty': 'hard',
                'description': 'Çelişkili durumları ele alabilmeli'
            },
        ]
    
    @staticmethod
    def get_performance_tests() -> List[Dict]:
        """Performans testleri (ölçeklenebilirlik için)"""
        return [
            {
                'id': 'perf_001',
                'name': 'Hız Testi - Basit Soru',
                'question': 'Bu belge ne?',
                'category': TestCategory.PERFORMANCE,
                'validators': ['Performance'],
                'expected_pattern': 'fast_response',
                'max_time': 5.0,  # 5 saniye
                'difficulty': 'easy',
                'description': 'Basit soru hızlı cevaplanmalı'
            },
            {
                'id': 'perf_002',
                'name': 'Hız Testi - Karmaşık Soru',
                'question': 'Bu belgede tüm olay ve fenomenler arasındaki ilişkiler nelerdir?',
                'category': TestCategory.PERFORMANCE,
                'validators': ['Performance'],
                'expected_pattern': 'complex_response_time',
                'max_time': 15.0,  # 15 saniye
                'difficulty': 'hard',
                'description': 'Karmaşık soru bile 15 saniyede cevaplandırılmalı'
            },
            {
                'id': 'perf_003',
                'name': 'Bellek Kullanımı',
                'question': 'Sorgu sayısı arttıkça sistem yavaşlıyor mu?',
                'category': TestCategory.PERFORMANCE,
                'validators': [],
                'expected_pattern': 'memory_stability',
                'difficulty': 'medium',
                'description': '100+ sorgu sonrası bellek sızıntısı olmamalı'
            },
        ]
    
    @staticmethod
    def get_robustness_tests() -> List[Dict]:
        """Sağlamlık testleri"""
        return [
            {
                'id': 'robust_001',
                'name': 'PDF Olmayan Format',
                'question': 'Sistem farklı PDF ler için konsisten sonuç mu veriyor?',
                'category': TestCategory.ROBUSTNESS,
                'validators': [],
                'expected_pattern': 'consistency',
                'difficulty': 'hard',
                'description': 'Farklı PDF lerde tutarlı davranmalı'
            },
            {
                'id': 'robust_002',
                'name': 'Eksik Veri Yönetimi',
                'question': 'Arızalı PDF (eksik sayfa vb.) ne olur?',
                'category': TestCategory.ROBUSTNESS,
                'validators': [],
                'expected_pattern': 'graceful_degradation',
                'difficulty': 'hard',
                'description': 'Arızalı veri bile düzgün hata vermelidanışy'
            },
            {
                'id': 'robust_003',
                'name': 'Çok Sayıda Dokuman',
                'question': '100 dokuman ile sistem stabil mi kalıyor?',
                'category': TestCategory.ROBUSTNESS,
                'validators': [],
                'expected_pattern': 'scalability',
                'difficulty': 'hard',
                'description': 'Ölçeklenebilirlik testi'
            },
        ]

# ===== TEST RAPORLAMA =====
class TestReport:
    """Test raporu oluşturma"""
    
    def __init__(self):
        self.results: List[TestResult] = []
    
    def add_result(self, result: TestResult):
        """Sonuç ekle"""
        self.results.append(result)
    
    def get_summary(self) -> Dict:
        """Özet istatistikler"""
        if not self.results:
            return {}
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        avg_similarity = sum(r.similarity_score for r in self.results) / total
        avg_time = sum(r.retrieval_time + r.llm_time for r in self.results) / total
        
        return {
            'total_tests': total,
            'passed': passed,
            'failed': total - passed,
            'pass_rate': f"{(passed/total)*100:.1f}%",
            'avg_similarity': f"{avg_similarity:.2f}",
            'avg_response_time': f"{avg_time:.2f}s",
            'by_category': self._group_by_category()
        }
    
    def _group_by_category(self) -> Dict:
        """Kategoriye göre gruplayın"""
        grouped = {}
        for result in self.results:
            cat = result.category.value
            if cat not in grouped:
                grouped[cat] = {'passed': 0, 'failed': 0}
            
            if result.passed:
                grouped[cat]['passed'] += 1
            else:
                grouped[cat]['failed'] += 1
        
        return grouped
    
    def export_to_json(self, filename: str):
        """JSON'a kaydet"""
        data = {
            'summary': self.get_summary(),
            'results': [
                {
                    'test_id': r.test_id,
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'similarity': r.similarity_score,
                    'response_time': r.retrieval_time + r.llm_time,
                    'category': r.category.value
                }
                for r in self.results
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def print_report(self):
        """Raporu yazdır"""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("TEST RAPORU")
        print("="*80)
        
        print(f"\nGenel Sonuçlar:")
        print(f"  Toplam Test: {summary.get('total_tests', 0)}")
        print(f"  Geçti: {summary.get('passed', 0)}")
        print(f"  Başarısız: {summary.get('failed', 0)}")
        print(f"  Başarı Oranı: {summary.get('pass_rate', 'N/A')}")
        
        print(f"\nPerformans:")
        print(f"  Ortalama Benzerlik: {summary.get('avg_similarity', 'N/A')}")
        print(f"  Ortalama Yanıt Süresi: {summary.get('avg_response_time', 'N/A')}")
        
        print(f"\nKategoriye Göre:")
        for category, stats in summary.get('by_category', {}).items():
            total = stats['passed'] + stats['failed']
            rate = (stats['passed'] / total * 100) if total > 0 else 0
            print(f"  {category}: {stats['passed']}/{total} (%{rate:.0f})")

# ===== ÖRNEK KULLANIM =====
if __name__ == "__main__":
    
    # Test suite oluştur
    suite = ComprehensiveTestSuite()
    
    # Tüm testleri al
    all_tests = (
        suite.get_basic_tests() +
        suite.get_complex_tests() +
        suite.get_hallucination_tests() +
        suite.get_edge_case_tests() +
        suite.get_performance_tests() +
        suite.get_robustness_tests()
    )
    
    print("="*80)
    print("PDF CHATBOT TEST SUITE")
    print("="*80)
    
    print(f"\nToplam {len(all_tests)} test case bulundu:\n")
    
    # Kategoriye göre göster
    by_cat = {}
    for test in all_tests:
        cat = test['category'].value
        by_cat[cat] = by_cat.get(cat, 0) + 1
    
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat}: {count} test")
    
    # Örnek rapor
    print("\n" + "-"*80)
    print("ÖRNEK TEST SONUÇLARI")
    print("-"*80)
    
    report = TestReport()
    
    # Mock results
    report.add_result(TestResult(
        test_name="Özet Soru",
        test_id="basic_001",
        category=TestCategory.BASIC_RETRIEVAL,
        question="Bu belge hakkında bilgi ver",
        answer="Bu belge yapay zeka hakkında ..."
,
        expected_pattern="summary",
        passed=True,
        similarity_score=0.85,
        retrieval_time=0.5,
        llm_time=2.1,
        retrieved_chunks=3,
        confidence_score=0.92
    ))
    
    report.print_report()
    report.export_to_json("test_results.json")
    
    print("\n✓ Test framework hazır!")
    print("İlgili PDF chatbot scriptleriyle birlikte kullanın.")
