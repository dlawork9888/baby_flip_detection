# baby_flip_detection
from Mediapipe
##### modules_test.ipynb에서 자유롭게 테스트해볼 수 있습니다.

### 모듈 불러와서 바로 사용하기
```python
import flip_detection_modules

# 내장 캠 사용
flip_detection_modules.flip_detection_cam()
# 비디오에서 flip 감지
flip_detection_modules.flip_detection_video('video_path')
```
flip이 감지 되었을 때 추가적인 로직이 필요한 경우, 
- flip_detection_modules.py
   - flip_detection_cam
   - flip_detection_video

위 두 함수의 "추가적인 로직 작성"을 편집해주세요!

