"""Closed RGB target profiles shared by language and perception."""
import numpy as np
import cv2

DESCRIPTIONS = ['气球', '红色气球', '普通红色气球', '黄色气球', '带白色横条的红色气球']

def instruction_target(text):
    if '气球' not in text:
        return None
    if any(word in text for word in ['蓝色', '绿色', '紫色', '黑色']):
        raise ValueError('UNSUPPORTED_BALLOON_COLOR')
    if '黄' in text and ('红' in text or '横条' in text or '白条' in text):
        raise ValueError('MULTIPLE_TARGET_DESCRIPTIONS')
    if '黄' in text:
        return '黄色气球'
    if '横条' in text or '白条' in text or '白带' in text:
        return '带白色横条的红色气球'
    if '红' in text:
        return '普通红色气球' if '普通' in text or '无白条' in text else '红色气球'
    return '气球'

def yellow_evidence(image, box, truncated=False):
    x1,y1,x2,y2=box
    if truncated or min(x2-x1,y2-y1)<24:
        return dict(matched=False, yellow_fraction=0., reason='BOX_SMALL_OR_TRUNCATED')
    h,w=image.shape[:2]
    # Central area reduces background contamination; source RGB only.
    a,b=int(max(0,x1+(x2-x1)*.2)),int(min(w,x2-(x2-x1)*.2))
    c,d=int(max(0,y1+(y2-y1)*.2)),int(min(h,y2-(y2-y1)*.2))
    roi=image[c:d,a:b]
    if not roi.size:
        return dict(matched=False, yellow_fraction=0., reason='EMPTY_ROI')
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    mask=(hsv[:,:,0]>=18)&(hsv[:,:,0]<=38)&(hsv[:,:,1]>=80)&(hsv[:,:,2]>=55)
    fraction=float(np.mean(mask))
    return dict(matched=fraction>=.25, yellow_fraction=fraction, reason='CURRENT_RGB_YELLOW_PROFILE',
                scope='engineering color threshold, not calibrated identity probability')
