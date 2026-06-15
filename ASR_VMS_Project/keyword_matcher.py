import re
import jellyfish
from rapidfuzz import fuzz

def normalize(text):
    """Normalize text: lowercase, strip punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

class KeywordMatcher:
    def __init__(self, keywords, fuzzy_ratio=85.0, conf_threshold=0.5):
        self.fuzzy_ratio = fuzzy_ratio
        self.conf_threshold = conf_threshold
        
        # Precompute target keyword features
        self.targets = []
        for kw in keywords:
            tokens = normalize(kw)
            if not tokens:
                continue
            metaphones = [jellyfish.metaphone(t) for t in tokens]
            self.targets.append({
                "raw": kw,
                "tokens": tokens,
                "length": len(tokens),
                "metaphones": metaphones
            })

    def non_max_suppression(self, predictions, window_s=1.0):
        if not predictions:
            return []
        predictions = sorted(predictions, key=lambda x: x["start"])
        clusters = []
        current_cluster = [predictions[0]]
        
        for p in predictions[1:]:
            if p["start"] - current_cluster[-1]["start"] <= window_s:
                current_cluster.append(p)
            else:
                clusters.append(current_cluster)
                current_cluster = [p]
        clusters.append(current_cluster)
        
        suppressed = []
        for cluster in clusters:
            # Pick highest confidence
            best_p = max(cluster, key=lambda x: x["confidence"])
            suppressed.append(best_p)
            
        return suppressed

    def find(self, words):
        """
        words: list of {"word": str, "start": float, "end": float, "score": float}
        Returns list of detections after NMS.
        """
        detections = []
        
        # Normalize incoming words
        norm_words = []
        for w in words:
            # We assume word might have punctuation, normalize it
            w_norm = normalize(w["word"])
            if w_norm:
                # If a whisper word splits into multiple, just take the first for phonetic simplicity
                # or keep it simple. Usually whisper words are single tokens.
                t = w_norm[0]
                norm_words.append({
                    "orig": w,
                    "token": t,
                    "metaphone": jellyfish.metaphone(t)
                })
        
        if not norm_words:
            return detections
            
        for target in self.targets:
            n = target["length"]
            if n == 0:
                continue
                
            # Sliding window of size n
            for i in range(len(norm_words) - n + 1):
                window = norm_words[i:i+n]
                
                # Check match
                exact_match = True
                phonetic_match = True
                avg_fuzzy = 0.0
                avg_score = 0.0
                
                for j in range(n):
                    t_token = target["tokens"][j]
                    t_meta = target["metaphones"][j]
                    w_token = window[j]["token"]
                    w_meta = window[j]["metaphone"]
                    
                    if t_token != w_token:
                        exact_match = False
                    if t_meta != w_meta:
                        phonetic_match = False
                        
                    avg_fuzzy += fuzz.ratio(t_token, w_token)
                    avg_score += window[j]["orig"]["score"]
                    
                avg_fuzzy /= n
                avg_score /= n
                
                match_type = None
                match_weight = 0.0
                
                if exact_match:
                    match_type = "exact"
                    match_weight = 1.0
                elif phonetic_match:
                    match_type = "phonetic"
                    match_weight = 0.9
                elif avg_fuzzy >= self.fuzzy_ratio:
                    match_type = "fuzzy"
                    match_weight = avg_fuzzy / 100.0
                    
                if match_type:
                    confidence = avg_score * match_weight
                    if confidence >= self.conf_threshold:
                        detections.append({
                            "keyword": target["raw"],
                            "start": window[0]["orig"]["start"],
                            "end": window[-1]["orig"]["end"],
                            "confidence": confidence,
                            "match_type": match_type,
                            "transcript_context": " ".join([w["orig"]["word"] for w in window])
                        })
                        
        # Apply NMS
        return self.non_max_suppression(detections)
