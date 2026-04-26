# Mental Health MBTI RAG API

نسخة API-only مهيأة للـ Swagger وتركّز على:
- FastAPI
- RAG على ملف MBTI + ملف Rufi
- فصل صحيح بين `video` / `podcast` / `book`
- تنظيف العناوين قبل العرض
- فلترة التوصيات بدون روابط عند تفعيل `recommendation_links_only`
- ربط MBTI بصياغة احتمالية غير تشخيصية
- Tone قابل للتحكم من الإعدادات أو من الطلب (`feminine` / `neutral`)

## أهم التحسينات في هذه النسخة
- `source_domain` أصبح hostname حقيقي للرابط مثل `www.youtube.com` أو `open.spotify.com`
- `source_collection` يوضح هل المورد من `mbti` أو `emotion`
- `recommended_podcasts` أضيفت للاستجابة
- الفيديوهات والبودكاست لا تظهر في الاستجابة إذا كانت بلا رابط عند تفعيل `recommendation_links_only=true`
- الكتب لم تعد تتضمن تلقائيًا فيديوهات من نوع "شرح كتاب" أو "تلخيص كتاب"
- الربط بـ MBTI أصبح بصياغة: "قد يساعد في فهم ميل عام" بدل تفسير قطعي

## التشغيل
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m scripts.build_index
uvicorn app.main:app --reload
```

Swagger:
- `http://127.0.0.1:8000/docs`

## أهم إعدادات `.env`
```env
TARGET_AUDIENCE=female_only
RESPONSE_STYLE=feminine
RECOMMENDATIONS_AFTER_TURN=3
ENABLE_GITHUB_CHAT_GENERATION=true
```

- `TARGET_AUDIENCE=female_only` أو `all`
- `RESPONSE_STYLE=feminine` أو `neutral`

## شكل الطلب الأساسي
```json
{
  "conversation_id": "demo-1",
  "mbti_type": "INFJ",
  "user_message": "أنا مستنزفة من الناس ومش بعرف أحط حدود",
  "user_gender": "female",
  "response_style": "config",
  "max_videos": 2,
  "max_books": 2,
  "max_podcasts": 2,
  "recommendation_links_only": true
}
```

## ملاحظات
- المشروع ليس أداة تشخيص نفسي.
- في حالات الخطر أو إيذاء النفس يتم إيقاف الرد العادي وتفعيل رد أمان.
- جودة التوصيات مرتبطة بوجود روابط نظيفة داخل المصدر نفسه؛ هذه النسخة تفضّل إخفاء المورد على إرجاع recommendation ضعيف أو بلا رابط.
