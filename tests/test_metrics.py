# Unit tests for retrieval metrics
"""
Tests para las métricas de evaluación de sistemas de recuperación.
"""

import unittest
from src.retriever.metrics import RecallAtK, PrecisionAtK, F1AtK, MRR, MAP, NDCG


class TestRecallAtK(unittest.TestCase):
    """Tests para la métrica Recall@k."""
    
    def test_recall_at_k_basic(self):
        """Test básico de Recall@k con algunos documentos relevantes recuperados."""
        metric = RecallAtK()
        retrieved = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7), ("doc4", 0.6)]
        relevant = ["doc1", "doc3", "doc5"]
        
        result = metric.compute(retrieved, relevant, k=2)
        
        self.assertEqual(result['score'], 1/3)  # 1 relevante de 3 totales
        self.assertEqual(result['metric_name'], 'Recall@2')
        self.assertEqual(result['k'], 2)
        self.assertEqual(result['relevant_found'], 1)
        self.assertEqual(result['total_relevant'], 3)
    
    def test_recall_perfect(self):
        """Test con recall perfecto (todos los relevantes encontrados)."""
        metric = RecallAtK()
        retrieved = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        relevant = ["doc1", "doc2"]
        
        result = metric.compute(retrieved, relevant, k=3)
        
        self.assertEqual(result['score'], 1.0)
        self.assertEqual(result['relevant_found'], 2)
        self.assertEqual(result['total_relevant'], 2)
    
    def test_recall_no_k(self):
        """Test sin especificar k (usa todos los documentos recuperados)."""
        metric = RecallAtK()
        retrieved = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        relevant = ["doc1", "doc3"]
        
        result = metric.compute(retrieved, relevant)
        
        self.assertEqual(result['score'], 1.0)  # 2 de 2 relevantes encontrados
        self.assertEqual(result['metric_name'], 'Recall')
    
    def test_recall_empty_relevant(self):
        """Test con lista vacía de documentos relevantes."""
        metric = RecallAtK()
        retrieved = [("doc1", 0.9), ("doc2", 0.8)]
        relevant = []
        
        result = metric.compute(retrieved, relevant, k=2)
        
        self.assertEqual(result['score'], 0.0)
        self.assertEqual(result['relevant_found'], 0)
        self.assertEqual(result['total_relevant'], 0)
    
    def test_recall_no_relevant_found(self):
        """Test donde no se encuentra ningún documento relevante."""
        metric = RecallAtK()
        retrieved = [("doc1", 0.9), ("doc2", 0.8)]
        relevant = ["doc3", "doc4"]
        
        result = metric.compute(retrieved, relevant, k=2)
        
        self.assertEqual(result['score'], 0.0)
        self.assertEqual(result['relevant_found'], 0)
        self.assertEqual(result['total_relevant'], 2)


class TestMRR(unittest.TestCase):
    """Tests para la métrica MRR (Mean Reciprocal Rank)."""
    
    def test_mrr_first_position(self):
        """Test con documento relevante en primera posición."""
        metric = MRR()
        retrieved = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        relevant = ["doc1", "doc4"]
        
        result = metric.compute(retrieved, relevant)
        
        self.assertEqual(result['score'], 1.0)
        self.assertEqual(result['first_relevant_rank'], 1)
    
    def test_mrr_second_position(self):
        """Test con primer documento relevante en segunda posición."""
        metric = MRR()
        retrieved = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        relevant = ["doc2", "doc4"]
        
        result = metric.compute(retrieved, relevant)
        
        self.assertEqual(result['score'], 0.5)
        self.assertEqual(result['first_relevant_rank'], 2)
    
    def test_mrr_third_position(self):
        """Test con primer documento relevante en tercera posición."""
        metric = MRR()
        retrieved = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        relevant = ["doc3", "doc4"]
        
        result = metric.compute(retrieved, relevant)
        
        self.assertAlmostEqual(result['score'], 1/3, places=5)
        self.assertEqual(result['first_relevant_rank'], 3)
    
    def test_mrr_no_relevant(self):
        """Test sin documentos relevantes encontrados."""
        metric = MRR()
        retrieved = [("doc1", 0.9), ("doc2", 0.8)]
        relevant = ["doc3", "doc4"]
        
        result = metric.compute(retrieved, relevant)
        
        self.assertEqual(result['score'], 0.0)
        self.assertIsNone(result['first_relevant_rank'])
    
    def test_mrr_empty_lists(self):
        """Test con listas vacías."""
        metric = MRR()
        
        # Documentos relevantes vacíos
        result1 = metric.compute([("doc1", 0.9)], [])
        self.assertEqual(result1['score'], 0.0)
        
        # Documentos recuperados vacíos
        result2 = metric.compute([], ["doc1"])
        self.assertEqual(result2['score'], 0.0)
    
    def test_mrr_with_k(self):
        """Test con k especificado (solo considera top-k)."""
        metric = MRR()
        retrieved = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7), ("doc4", 0.6)]
        relevant = ["doc3"]
        
        # Con k=2, no encuentra doc3 (está en posición 3)
        result1 = metric.compute(retrieved, relevant, k=2)
        self.assertEqual(result1['score'], 0.0)
        
        # Con k=3, sí encuentra doc3
        result2 = metric.compute(retrieved, relevant, k=3)
        self.assertAlmostEqual(result2['score'], 1/3, places=5)


class TestMAP(unittest.TestCase):
    """Tests para la métrica MAP (Mean Average Precision)."""
    
    def test_map_basic(self):
        """Test básico de MAP."""
        metric = MAP()
        # Posiciones: doc1(1), doc2(2-REL), doc3(3-REL), doc4(4)
        retrieved = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7), ("doc4", 0.6)]
        relevant = ["doc2", "doc3"]
        
        result = metric.compute(retrieved, relevant)
        
        # Precisiones en posiciones relevantes:
        # Posición 2 (doc2): 1/2 = 0.5
        # Posición 3 (doc3): 2/3 ≈ 0.667
        # AP = (0.5 + 0.667) / 2 = 0.583
        expected_ap = (0.5 + 2/3) / 2
        self.assertAlmostEqual(result['score'], expected_ap, places=3)
        self.assertEqual(result['num_relevant_retrieved'], 2)
        self.assertEqual(len(result['precisions_at_relevant']), 2)
    
    def test_map_perfect_ranking(self):
        """Test con ranking perfecto (todos relevantes al inicio)."""
        metric = MAP()
        retrieved = [("doc1", 1.0), ("doc2", 0.9), ("doc3", 0.5)]
        relevant = ["doc1", "doc2"]
        
        result = metric.compute(retrieved, relevant)
        
        # Precisiones: P@1 = 1/1 = 1.0, P@2 = 2/2 = 1.0
        # AP = (1.0 + 1.0) / 2 = 1.0
        self.assertEqual(result['score'], 1.0)
    
    def test_map_no_relevant_found(self):
        """Test sin documentos relevantes encontrados."""
        metric = MAP()
        retrieved = [("doc1", 0.9), ("doc2", 0.8)]
        relevant = ["doc3", "doc4"]
        
        result = metric.compute(retrieved, relevant)
        
        self.assertEqual(result['score'], 0.0)
        self.assertEqual(result['num_relevant_retrieved'], 0)
        self.assertEqual(len(result['precisions_at_relevant']), 0)
    
    def test_map_at_k(self):
        """Test MAP@k (solo considera top-k documentos)."""
        metric = MAP()
        retrieved = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7), ("doc4", 0.6)]
        relevant = ["doc2", "doc4"]
        
        # MAP@2: solo ve doc1 y doc2
        result = metric.compute(retrieved, relevant, k=2)
        
        # Solo doc2 está en top-2, en posición 2
        # AP = (1/2) / 2 = 0.25
        self.assertAlmostEqual(result['score'], 0.25, places=3)
        self.assertEqual(result['metric_name'], 'MAP@2')
    
    def test_map_empty_relevant(self):
        """Test con lista vacía de relevantes."""
        metric = MAP()
        retrieved = [("doc1", 0.9), ("doc2", 0.8)]
        relevant = []
        
        result = metric.compute(retrieved, relevant)
        
        self.assertEqual(result['score'], 0.0)


class TestNDCG(unittest.TestCase):
    """Tests para la métrica nDCG (Normalized Discounted Cumulative Gain)."""
    
    def test_ndcg_perfect_ranking(self):
        """Test con ranking perfecto (todos relevantes primero)."""
        metric = NDCG()
        retrieved = [("doc1", 1.0), ("doc2", 0.9), ("doc3", 0.5)]
        relevant = ["doc1", "doc2"]
        
        result = metric.compute(retrieved, relevant, k=3)
        
        # Ranking perfecto: nDCG debe ser 1.0
        self.assertEqual(result['score'], 1.0)
        self.assertEqual(result['metric_name'], 'nDCG@3')
    
    def test_ndcg_imperfect_ranking(self):
        """Test con ranking imperfecto."""
        metric = NDCG()
        # doc2 es relevante pero está en posición 2
        retrieved = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        relevant = ["doc2"]
        
        result = metric.compute(retrieved, relevant, k=3)
        
        # DCG: (2^0 - 1)/log2(2) + (2^1 - 1)/log2(3) + (2^0 - 1)/log2(4)
        #    = 0 + 1/log2(3) + 0
        # IDCG: (2^1 - 1)/log2(2) = 1/1 = 1.0
        # nDCG = DCG / IDCG
        import numpy as np
        expected_dcg = 1 / np.log2(3)
        expected_idcg = 1.0
        expected_ndcg = expected_dcg / expected_idcg
        
        self.assertAlmostEqual(result['score'], expected_ndcg, places=5)
        self.assertGreater(result['score'], 0.0)
        self.assertLess(result['score'], 1.0)
    
    def test_ndcg_no_relevant(self):
        """Test sin documentos relevantes encontrados."""
        metric = NDCG()
        retrieved = [("doc1", 0.9), ("doc2", 0.8)]
        relevant = ["doc3", "doc4"]
        
        result = metric.compute(retrieved, relevant, k=2)
        
        self.assertEqual(result['score'], 0.0)
    
    def test_ndcg_empty_relevant(self):
        """Test con lista vacía de relevantes."""
        metric = NDCG()
        retrieved = [("doc1", 0.9), ("doc2", 0.8)]
        relevant = []
        
        result = metric.compute(retrieved, relevant, k=2)
        
        self.assertEqual(result['score'], 0.0)
        self.assertEqual(result['dcg'], 0.0)
        self.assertEqual(result['idcg'], 0.0)
    
    def test_ndcg_multiple_relevant(self):
        """Test con múltiples documentos relevantes."""
        metric = NDCG()
        retrieved = [("doc1", 1.0), ("doc2", 0.9), ("doc3", 0.8), ("doc4", 0.7)]
        relevant = ["doc1", "doc3", "doc5"]
        
        result = metric.compute(retrieved, relevant, k=4)
        
        # doc1 en pos 1 (relevante), doc3 en pos 3 (relevante)
        # DCG > 0, pero < IDCG (no es perfecto)
        self.assertGreater(result['score'], 0.0)
        self.assertLess(result['score'], 1.0)
        self.assertGreater(result['dcg'], 0.0)
        self.assertGreater(result['idcg'], 0.0)
    
    def test_ndcg_no_k(self):
        """Test sin especificar k (usa todos los documentos)."""
        metric = NDCG()
        retrieved = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        relevant = ["doc1"]
        
        result = metric.compute(retrieved, relevant)
        
        self.assertEqual(result['k'], 3)
        self.assertEqual(result['score'], 1.0)  # Perfecto porque doc1 está primero


class TestPrecisionAtK(unittest.TestCase):
    """Tests para la métrica Precision@k."""
    
    def test_precision_at_k_basic(self):
        """Test básico de Precision@k con algunos documentos relevantes."""
        metric = PrecisionAtK()
        retrieved = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7), ("doc4", 0.6)]
        relevant = ["doc1", "doc3", "doc5"]
        
        result = metric.compute(retrieved, relevant, k=3)
        
        # 2 relevantes (doc1, doc3) de 3 recuperados = 2/3
        self.assertAlmostEqual(result['score'], 2/3, places=5)
        self.assertEqual(result['metric_name'], 'Precision@3')
        self.assertEqual(result['k'], 3)
        self.assertEqual(result['relevant_found'], 2)
        self.assertEqual(result['total_retrieved'], 3)
    
    def test_precision_perfect(self):
        """Test con precision perfecta (todos recuperados son relevantes)."""
        metric = PrecisionAtK()
        retrieved = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        relevant = ["doc1", "doc2", "doc3", "doc4"]
        
        result = metric.compute(retrieved, relevant, k=3)
        
        self.assertEqual(result['score'], 1.0)
        self.assertEqual(result['relevant_found'], 3)
        self.assertEqual(result['total_retrieved'], 3)
    
    def test_precision_no_relevant(self):
        """Test sin documentos relevantes en top-k."""
        metric = PrecisionAtK()
        retrieved = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        relevant = ["doc4", "doc5"]
        
        result = metric.compute(retrieved, relevant, k=3)
        
        self.assertEqual(result['score'], 0.0)
        self.assertEqual(result['relevant_found'], 0)
    
    def test_precision_no_k(self):
        """Test sin especificar k (usa todos los documentos recuperados)."""
        metric = PrecisionAtK()
        retrieved = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        relevant = ["doc1", "doc3"]
        
        result = metric.compute(retrieved, relevant)
        
        self.assertAlmostEqual(result['score'], 2/3, places=5)
        self.assertEqual(result['metric_name'], 'Precision')
    
    def test_precision_empty_relevant(self):
        """Test con lista vacía de documentos relevantes."""
        metric = PrecisionAtK()
        retrieved = [("doc1", 0.9), ("doc2", 0.8)]
        relevant = []
        
        result = metric.compute(retrieved, relevant, k=2)
        
        self.assertEqual(result['score'], 0.0)
        self.assertEqual(result['relevant_found'], 0)
    
    def test_precision_empty_retrieved(self):
        """Test con lista vacía de documentos recuperados."""
        metric = PrecisionAtK()
        retrieved = []
        relevant = ["doc1", "doc2"]
        
        result = metric.compute(retrieved, relevant, k=2)
        
        self.assertEqual(result['score'], 0.0)
        self.assertEqual(result['total_retrieved'], 0)
    
    def test_precision_k_greater_than_retrieved(self):
        """Test donde k es mayor que documentos disponibles."""
        metric = PrecisionAtK()
        retrieved = [("doc1", 0.9), ("doc2", 0.8)]
        relevant = ["doc1"]
        
        result = metric.compute(retrieved, relevant, k=10)
        
        # Usa solo los 2 disponibles
        self.assertAlmostEqual(result['score'], 1/2, places=5)
        self.assertEqual(result['k'], 2)
        self.assertEqual(result['total_retrieved'], 2)


class TestF1AtK(unittest.TestCase):
    """Tests para la métrica F1@k."""
    
    def test_f1_basic(self):
        """Test básico de F1@k."""
        metric = F1AtK()
        retrieved = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7), ("doc4", 0.6)]
        relevant = ["doc1", "doc3", "doc5"]
        
        result = metric.compute(retrieved, relevant, k=3)
        
        # Precision@3 = 2/3, Recall@3 = 2/3
        # F1 = 2 * (2/3 * 2/3) / (2/3 + 2/3) = 2/3
        self.assertAlmostEqual(result['score'], 2/3, places=5)
        self.assertAlmostEqual(result['precision'], 2/3, places=5)
        self.assertAlmostEqual(result['recall'], 2/3, places=5)
        self.assertEqual(result['relevant_found'], 2)
    
    def test_f1_perfect(self):
        """Test con F1 perfecto (P=1, R=1)."""
        metric = F1AtK()
        retrieved = [("doc1", 0.9), ("doc2", 0.8)]
        relevant = ["doc1", "doc2"]
        
        result = metric.compute(retrieved, relevant, k=2)
        
        self.assertEqual(result['score'], 1.0)
        self.assertEqual(result['precision'], 1.0)
        self.assertEqual(result['recall'], 1.0)
    
    def test_f1_high_precision_low_recall(self):
        """Test con alta precision pero bajo recall."""
        metric = F1AtK()
        retrieved = [("doc1", 0.9), ("doc2", 0.8)]
        relevant = ["doc1", "doc3", "doc4", "doc5"]  # Solo 1 de 4 recuperado
        
        result = metric.compute(retrieved, relevant, k=2)
        
        # Precision = 1/2 = 0.5, Recall = 1/4 = 0.25
        # F1 = 2 * (0.5 * 0.25) / (0.5 + 0.25) = 0.25 / 0.75 = 1/3
        expected_p = 1/2
        expected_r = 1/4
        expected_f1 = 2 * (expected_p * expected_r) / (expected_p + expected_r)
        
        self.assertAlmostEqual(result['precision'], expected_p, places=5)
        self.assertAlmostEqual(result['recall'], expected_r, places=5)
        self.assertAlmostEqual(result['score'], expected_f1, places=5)
    
    def test_f1_low_precision_high_recall(self):
        """Test con baja precision pero alto recall."""
        metric = F1AtK()
        retrieved = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7), ("doc4", 0.6)]
        relevant = ["doc1", "doc2"]  # Solo 2 de 4 son relevantes
        
        result = metric.compute(retrieved, relevant, k=4)
        
        # Precision = 2/4 = 0.5, Recall = 2/2 = 1.0
        # F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 1.0 / 1.5 = 2/3
        expected_p = 2/4
        expected_r = 1.0
        expected_f1 = 2 * (expected_p * expected_r) / (expected_p + expected_r)
        
        self.assertAlmostEqual(result['precision'], expected_p, places=5)
        self.assertAlmostEqual(result['recall'], expected_r, places=5)
        self.assertAlmostEqual(result['score'], expected_f1, places=5)
    
    def test_f1_zero_division(self):
        """Test con casos donde P o R es 0."""
        metric = F1AtK()
        
        # Ningún relevante recuperado
        retrieved1 = [("doc1", 0.9), ("doc2", 0.8)]
        relevant1 = ["doc3", "doc4"]
        result1 = metric.compute(retrieved1, relevant1, k=2)
        
        self.assertEqual(result1['score'], 0.0)
        self.assertEqual(result1['precision'], 0.0)
        self.assertEqual(result1['recall'], 0.0)
        
        # Lista vacía de relevantes
        result2 = metric.compute([("doc1", 0.9)], [], k=1)
        self.assertEqual(result2['score'], 0.0)
    
    def test_f1_compare_with_precision_recall(self):
        """Verificar que F1 metadata coincide con Precision y Recall calculados."""
        metric_f1 = F1AtK()
        metric_p = PrecisionAtK()
        metric_r = RecallAtK()
        
        retrieved = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        relevant = ["doc1", "doc3", "doc5"]
        k = 3
        
        result_f1 = metric_f1.compute(retrieved, relevant, k=k)
        result_p = metric_p.compute(retrieved, relevant, k=k)
        result_r = metric_r.compute(retrieved, relevant, k=k)
        
        # Verificar que los valores coinciden
        self.assertAlmostEqual(result_f1['precision'], result_p['score'], places=5)
        self.assertAlmostEqual(result_f1['recall'], result_r['score'], places=5)
        
        # Verificar fórmula F1
        expected_f1 = 2 * (result_p['score'] * result_r['score']) / (result_p['score'] + result_r['score'])
        self.assertAlmostEqual(result_f1['score'], expected_f1, places=5)
    
    def test_f1_no_k(self):
        """Test sin especificar k."""
        metric = F1AtK()
        retrieved = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        relevant = ["doc1", "doc2"]
        
        result = metric.compute(retrieved, relevant)
        
        # P = 2/3, R = 2/2 = 1.0
        # F1 = 2 * (2/3 * 1) / (2/3 + 1) = 4/3 / 5/3 = 4/5
        expected_f1 = 2 * (2/3 * 1.0) / (2/3 + 1.0)
        self.assertAlmostEqual(result['score'], expected_f1, places=5)
        self.assertEqual(result['metric_name'], 'F1')


class TestMetricInterface(unittest.TestCase):
    """Tests para verificar que todas las métricas implementan la interfaz correctamente."""
    
    def test_all_metrics_have_name_property(self):
        """Verifica que todas las métricas tienen la propiedad name."""
        metrics = [RecallAtK(), PrecisionAtK(), F1AtK(), MRR(), MAP(), NDCG()]
        
        for metric in metrics:
            self.assertTrue(hasattr(metric, 'name'))
            self.assertIsInstance(metric.name, str)
            self.assertGreater(len(metric.name), 0)
    
    def test_all_metrics_return_dict(self):
        """Verifica que todas las métricas retornan un diccionario."""
        metrics = [RecallAtK(), PrecisionAtK(), F1AtK(), MRR(), MAP(), NDCG()]
        retrieved = [("doc1", 0.9), ("doc2", 0.8)]
        relevant = ["doc1"]
        
        for metric in metrics:
            result = metric.compute(retrieved, relevant, k=2)
            self.assertIsInstance(result, dict)
            self.assertIn('score', result)
            self.assertIn('metric_name', result)
            self.assertIsInstance(result['score'], float)
    
    def test_all_metrics_score_in_valid_range(self):
        """Verifica que todas las métricas retornan scores en rango válido [0, 1]."""
        metrics = [RecallAtK(), PrecisionAtK(), F1AtK(), MRR(), MAP(), NDCG()]
        retrieved = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        relevant = ["doc2"]
        
        for metric in metrics:
            result = metric.compute(retrieved, relevant, k=3)
            self.assertGreaterEqual(result['score'], 0.0)
            self.assertLessEqual(result['score'], 1.0)


if __name__ == '__main__':
    unittest.main()
