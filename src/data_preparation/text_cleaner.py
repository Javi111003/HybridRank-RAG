# src/data_preparation/text_cleaner.py

import re
import unicodedata
from typing import List, Dict, Any, Optional

import spacy

try:
    nlp_model = spacy.load("es_core_news_md")
    custom_stopwords = {
        "ley", "decreto", "resolución", "artículo", "inciso", "apartado", "literal",
        "número", "año", "gaceta", "oficial", "cuba", "cubano", "cubana", "república",
        "ministerio", "consejo", "estado", "gobierno", "poder", "popular", "nacional"
    }
    nlp_model.Defaults.stop_words.update(custom_stopwords)
except Exception as e:
    nlp_model = None
    print(f"No se pudo cargar el modelo de SpaCy 'es_core_news_md': {e}")
    print("La lematización y el filtrado de stopwords no se aplicarán en TextCleaner.")


class TextCleaner:
    """
    Clase para limpiar y normalizar texto extraído de documentos legales cubanos (Gaceta Oficial).
    Aplica una serie de operaciones de pre-procesamiento adaptadas para textos jurídicos
    para su posterior segmentación y generación de embeddings.
    Trabaja con la salida estructurada del DocumentLoader (lista de diccionarios).
    """
    def __init__(self):
        """
        Inicializa el limpiador de texto. El modelo de SpaCy se carga una vez a nivel de módulo.
        """
        self.nlp = nlp_model

    def extract_legal_references(self, text: str):
        """
        Extrae referencias legales (artículos, incisos, etc.) para preservarlas durante la limpieza.
        """
        pattern = r"""
            (                                           # Grupo principal
            (?:Art[íi]culo|Art\.|ARTÍCULO)\s*\d+[a-z]?  # Artículo 123, Art. 45a
            |(?:Inciso|INCISO)\s*[a-z]\)               # Inciso a), INCISO b)
            |(?:Apartado|APARTADO)\s*\d+               # Apartado 1, APARTADO 2
            |(?:Ley|LEY)\s*(?:No\.|Número|N[úu]mero)?\s*\d+(?:/\d+)?  # Ley No. 123/2019
            |(?:Decreto|DECRETO)(?:-Ley|-LEY)?\s*(?:No\.|Número)?\s*\d+(?:/\d+)?  # Decreto-Ley 15/2020
            |(?:Resolución|RESOLUCIÓN)\s*(?:No\.|Número)?\s*\d+(?:/\d+)?  # Resolución No. 45/2021
            |GOC-\d+-\w+-EX\d+                        # Código de Gaceta: GOC-2020-48-EX5
            )
        """
        references = re.findall(pattern, text, re.VERBOSE | re.IGNORECASE)
        
        text_sin_referencias = text
        for i, ref in enumerate(references):
            text_sin_referencias = text_sin_referencias.replace(ref, f"__legal_ref_{i}__")
        return text_sin_referencias, references
    
    def reincorporate_legal_references(self, text: str, references: List[str]) -> str:
        """Reincorpora las referencias legales al texto limpio."""
        for i, ref in enumerate(references):
            placeholder = f"__legal_ref_{i}__"
            text = text.replace(placeholder, ref)
        return text

    def remove_extra_whitespace(self, text: str) -> str:
        """Elimina espacios en blanco adicionales."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def normalize_unicode(self, text: str) -> str:
        """Normaliza caracteres Unicode manteniendo acentos españoles."""
        return unicodedata.normalize('NFKC', text)

    def remove_gaceta_headers_footers(self, text: str) -> str:
        """
        Elimina patrones comunes de encabezados y pies de página de la Gaceta Oficial.
        """
        patterns = [
            # Información de Gaceta
            r'Gaceta Oficial No\.\s*\d+.*?de.*?\d{4}',
            r'EXTRAORDINARIA AÑO C\w+',
            r'LA HABANA,.*?\d{4}',
            r'Sitio Web: http://www\.gacetaoficial\.gob\.cu/',
            r'Calle Zanja No\.\s*\d+.*?Centro Habana',
            r'Teléfonos:\s*[\d\-\s]+', 
            # Números de página
            r'^\s*Página\s+\d+\s*$',
            r'^\s*\d+\s*$',  # Solo números
            r'^\s*\[\s*\d+\s*\]\s*$',  # [123]
            # Códigos de referencia
            r'^\s*GOC-\d+-\w+-EX\d+\s*$',
            # Separadores y líneas
            r'^_{3,}$',  
            r'^-+$'
        ]

        cleaned_text = text
        for pattern in patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
        return cleaned_text

    def remove_urls(self, text: str) -> str:
        """Elimina URLs del texto."""
        return re.sub(r'(https?://[^\s]+)|(www\.[^\s]+)', '', text)

    def remove_emails(self, text: str) -> str:
        """Elimina direcciones de email del texto."""
        return re.sub(r'\S*@\S*\s?', '', text)

    def normalize_legal_formatting(self, text: str) -> str:
        """
        Normaliza el formato de textos legales cubanos.
        """
        # Normalizar "POR CUANTO:" y similares
        text = re.sub(r'POR\s+CUANTO\s*:', 'POR CUANTO:', text, flags=re.IGNORECASE)
        text = re.sub(r'CONSIDERANDO\s*:', 'CONSIDERANDO:', text, flags=re.IGNORECASE)
        text = re.sub(r'RESUELVO\s*:', 'RESUELVO:', text, flags=re.IGNORECASE)
        text = re.sub(r'DISPONGO\s*:', 'DISPONGO:', text, flags=re.IGNORECASE)
        
        # Normalizar referencias a artículos
        text = re.sub(r'Art[íi]culo\s+(\d+)', r'Artículo \1', text, flags=re.IGNORECASE)
        text = re.sub(r'Art\.\s*(\d+)', r'Artículo \1', text, flags=re.IGNORECASE)
        
        return text

    def remove_non_alphanumeric(self, text: str, keep_punctuation: str = r'.,!?;:()[]{}%-/°ª_') -> str:
        """
        Elimina caracteres no alfanuméricos excepto puntuación específica para textos legales.
        """
        pattern = r'[^\w\s' + re.escape(keep_punctuation) + r']'
        text = re.sub(pattern, '', text)
        return text

    def convert_to_lowercase(self, text: str) -> str:
        """Convierte texto a minúsculas."""
        return text.lower()

    def remove_stopwords_and_lemmatize(self, text: str, include_stopwords: bool = False) -> str:
        """
        Elimina stopwords y aplica lematización usando spaCy.
        Para textos legales, conserva términos técnicos importantes.
        """
        if not self.nlp:
            return text

        doc = self.nlp(text)
        tokens = []
        
        # Términos legales importantes que NO deben eliminarse como stopwords
        legal_terms_to_keep = {
            'ley', 'decreto', 'artículo', 'inciso', 'apartado', 'resolución',
            'ordenanza', 'disposición', 'reglamento', 'norma', 'código',
            'constitución', 'derecho', 'deber', 'obligación', 'responsabilidad'
        }
        
        for token in doc:
            if not token.is_space and not token.is_punct:
                token_lower = token.text.lower()
                # Incluir si no es stopword O si es un término legal importante
                if (include_stopwords or 
                    not token.is_stop or 
                    token_lower in legal_terms_to_keep or
                    token.pos_ in ['PROPN', 'NUM']):  # Mantener nombres propios y números
                    tokens.append(token.lemma_.lower())
        
        return " ".join(tokens)

    def clean_element_content(self, content: str, element_type: str, apply_lemmatization: bool = False) -> str:
        """
        Aplica limpieza específica según el tipo de elemento de documentos legales.

        :param content: El texto en bruto del elemento a limpiar.
        :param element_type: El tipo de elemento (ej., 'CompositeElement', 'Table', 'Title').
        :param apply_lemmatization: Si es True, aplica lematización y elimina stopwords.
        """
        if not content or not content.strip():
            return ""
            
        legal_references = []
        cleaned_content = content
        
        cleaned_content = self.remove_urls(cleaned_content)
        cleaned_content = self.remove_emails(cleaned_content)
        
        # Extraer referencias legales antes de limpiar
        cleaned_content, legal_references = self.extract_legal_references(cleaned_content)
        if element_type == 'Table':
            cleaned_content = self.remove_extra_whitespace(cleaned_content)
        else:
            cleaned_content = self.remove_gaceta_headers_footers(cleaned_content)
            cleaned_content = self.normalize_legal_formatting(cleaned_content)
            cleaned_content = self.normalize_unicode(cleaned_content)
            cleaned_content = self.remove_extra_whitespace(cleaned_content)
            
            if not self._is_important_header(cleaned_content):
                cleaned_content = self.remove_non_alphanumeric(
                    cleaned_content, 
                    keep_punctuation=r'.,!?;:()[]%-/°ªº_'
                )
            
            if legal_references:
                cleaned_content = self.reincorporate_legal_references(cleaned_content, legal_references)
                
            if apply_lemmatization:
                cleaned_content = self.remove_stopwords_and_lemmatize(cleaned_content)
            else:
                cleaned_content = self.convert_to_lowercase(cleaned_content)
        
        cleaned_content = self.remove_extra_whitespace(cleaned_content)
        
        return cleaned_content

    def _is_important_header(self, text: str) -> bool:
        """
        Determina si el texto es un encabezado importante que debe preservarse.
        """
        important_patterns = [
            r'^(?:LEY|DECRETO|RESOLUCIÓN|ORDENANZA)',
            r'^(?:ARTÍCULO|CAPÍTULO|SECCIÓN|TÍTULO)',
            r'^(?:POR CUANTO|CONSIDERANDO|RESUELVO|DISPONGO)',
            r'^\w+(?:\s+\w+){0,3}:$',  # Patrones como "HAGO SABER:"
        ]
        
        for pattern in important_patterns:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                return True
        return False

    def clean_documents(self, elements: List[Dict[str, Any]], apply_lemmatization: bool = False) -> List[Dict[str, Any]]:
        """
        Limpia una lista de elementos de documento estructurados (salida de DocumentLoader).
        Optimizado para documentos legales cubanos de la Gaceta Oficial.

        :param elements: Lista de diccionarios con elementos extraídos por Unstructured.io
        :param apply_lemmatization: Si es True, aplica lematización y elimina stopwords
        :return: Lista de elementos con contenido limpio
        """
        cleaned_elements: List[Dict[str, Any]] = []
        
        print(f"🧹 Iniciando limpieza de {len(elements)} elementos...")
        
        for i, element in enumerate(elements):
            try:
                original_content = element.get("content", "")
                element_type = element.get("metadata", {}).get("type", "CompositeElement")
                
                cleaned_content = self.clean_element_content(
                    original_content, 
                    element_type, 
                    apply_lemmatization
                )
                
                if cleaned_content and len(cleaned_content.strip()) > 10:
                    cleaned_element = element.copy()
                    cleaned_element["content"] = original_content
                    cleaned_element["cleaned_content"] = cleaned_content
                    cleaned_elements.append(cleaned_element)
                    
            except Exception as e:
                print(f"⚠️ Error limpiando elemento {i}: {e}")
                continue
        
        print(f"✅ Limpieza completada: {len(cleaned_elements)} elementos válidos de {len(elements)} originales")
        return cleaned_elements


if __name__ == "__main__":
    import os
    import json
    
    # Rutas de archivos
    current_script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
    input_file = os.path.join(project_root, '.data', 'loader_output', 'all_raw_extracted_elements.json')
    output_dir = os.path.join(project_root, '.data', 'cleaned_content')
    output_file = os.path.join(output_dir, 'cleaned_elements.json')
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar elementos desde el archivo JSON
    try:
        print(f"📂 Cargando elementos desde: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            elements = json.load(f)
        print(f"✅ Cargados {len(elements)} elementos")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {input_file}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error al leer JSON: {e}")
        exit(1)
    
    # Inicializar el limpiador
    cleaner = TextCleaner()
    
    # Limpiar documentos (sin lematización por defecto)
    print("\n🧹 Procesando elementos...")
    cleaned_elements = cleaner.clean_documents(elements, apply_lemmatization=False)
    
    # Guardar elementos limpios
    try:
        print(f"💾 Guardando elementos limpios en: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_elements, f, ensure_ascii=False, indent=2)
        print(f"✅ Guardados {len(cleaned_elements)} elementos limpios")
    except Exception as e:
        print(f"❌ Error al guardar archivo: {e}")
        exit(1)
            
#    # Opcional: También generar versión con lematización
#    output_file_lemma = os.path.join(output_dir, "cleaned_elements_lemmatized.json")
#    try:
#        print(f"\n🔤 Procesando con lematización...")
#        cleaned_elements_lemma = cleaner.clean_documents(elements, apply_lemmatization=True)
#        
#        print(f"💾 Guardando versión lematizada en: {output_file_lemma}")
#        with open(output_file_lemma, 'w', encoding='utf-8') as f:
#            json.dump(cleaned_elements_lemma, f, ensure_ascii=False, indent=2)
#        print(f"✅ Guardados {len(cleaned_elements_lemma)} elementos lematizados")
#    except Exception as e:
#        print(f"⚠️ Error al generar versión lematizada: {e}")
#    
#    print("\n📊 Resumen del procesamiento:")
#    print(f"   • Elementos originales: {len(elements)}")
#    print(f"   • Elementos limpios: {len(cleaned_elements)}")
#    print(f"   • Archivos generados:")
#    print(f"     - {output_file}")
#    if 'cleaned_elements_lemma' in locals():
#        print(f"     - {output_file_lemma}")