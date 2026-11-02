"""Answer-free output alerts; engineering thresholds, not paper hyperparameters."""
import re
from collections import Counter

WINDOW=100
EXCESS_RATE=.05

def repeated(text):
    tokens=re.findall(r'\w+|[^\w\s]',text.lower())
    if len(tokens)<32:
        return False
    grams=[tuple(tokens[i:i+4]) for i in range(len(tokens)-3)]
    return 1.-len(set(grams))/len(grams)>=.5

def quality_window(predictions,source):
    if len(predictions)<WINDOW:
        return {'status':'WAIT','n':len(predictions),'required':WINDOW}
    start=len(predictions)-WINDOW
    rows=predictions[start:]
    baseline=source[start:len(predictions)]
    if len(baseline)!=WINDOW or any(row['source_index']!=old['source_index'] for row,old in zip(rows,baseline)):
        raise ValueError('Quality comparison requires matched Source indexes')
    def count(values):
        return {'empty':sum(not row['predict'].strip() for row in values),
                'repetition':sum(repeated(row['predict']) for row in values)}
    current,original=count(rows),count(baseline)
    alerts=[key for key in current if current[key]-original[key]>EXCESS_RATE*WINDOW]
    texts=Counter(' '.join(row['predict'].lower().split()) for row in rows if row['predict'].strip())
    source_texts=Counter(' '.join(row['predict'].lower().split()) for row in baseline if row['predict'].strip())
    answer,amount=texts.most_common(1)[0] if texts else ('',0)
    old_amount=source_texts[answer]
    if amount>=.5*WINDOW and amount-old_amount>.1*WINDOW:
        alerts.append('dominant_answer')
    return {'status':'ALERT' if alerts else 'PASS','window_start':start,'n':WINDOW,
            'tlm':current,'source':original,'excess_rate_threshold':EXCESS_RATE,'alerts':alerts,
            'dominant_answer':{'tlm_count':amount,'source_count':old_amount,'text':answer[:180],
                               'share_threshold':.5,'excess_rate_threshold':.1}}
