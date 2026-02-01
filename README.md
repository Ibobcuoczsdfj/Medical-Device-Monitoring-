import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import datetime

# 1. إنشاء بيانات تجريبية (Simulated Data)
# المعايير: درجة الحرارة، ساعات التشغيل، عدد مرات الاستخدام، آخر صيانة
data = {
    'device_id': [101, 102, 103, 104, 105],
    'temperature': [35.5, 42.0, 37.0, 45.5, 36.2], # درجة الحرارة C
    'operating_hours': [1200, 3500, 800, 5000, 1500], # ساعات التشغيل
    'usage_frequency': [10, 50, 5, 80, 12], # الاستخدام اليومي
    'failure_risk': [0, 1, 0, 1, 0] # 1 تعني يحتاج صيانة، 0 تعني سليم
}

df = pd.DataFrame(data)

# 2. وظيفة حساب "مؤشر الصحة" (Health Score)
def calculate_health_score(temp, hours, usage):
    # معادلة بسيطة: تبدأ من 100 وتخصم نقاط بناءً على الإجهاد
    score = 100
    if temp > 40: score -= 20
    if hours > 3000: score -= 30
    if usage > 60: score -= 15
    return max(score, 0)

# 3. تدريب نموذج ذكاء اصطناعي بسيط (Machine Learning)
X = df[['temperature', 'operating_hours', 'usage_frequency']]
y = df['failure_risk']

model = RandomForestClassifier()
model.fit(X, y)

# 4. تجربة النظام على جهاز جديد
new_device_data = [[43.5, 4000, 75]] # جهاز حرارته عالية وساعاته كثيرة
prediction = model.predict(new_device_data)
health_score = calculate_health_score(43.5, 4000, 75)

print("--- Smart Medical Monitoring Platform ---")
print(f"Device Health Score: {health_score}/100")

if prediction[0] == 1 or health_score < 50:
    print("⚠️ Alert: High Failure Risk Predicted! Maintenance Required.")
else:
    print("✅ Device Status: Normal.")
 
