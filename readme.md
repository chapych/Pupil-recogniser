**Распознаватель зрачков на изображении**

Реализован собственный алгоритм, основанный на алгоритме Swirski (Robust real-time pupil tracking in highly off-axis images Lech Swirski). Работающий алгоритм был протестирован, найдена его точность ACC = 76%. 
В дальнейшей работе планируется усовершенствование алгоритма нахождения контура зрачка без использование встроенных библиотек: это позволит быстрее находить параметры эллипса, что критично для последующих этапов реализации eye-tracker`а. В планах реализация собственного Appearance-based алгоритма нахождения направления взгляда. Текущая работа будет основой будущего алгоритма. 

*Пример работы алгоритма на датасетах Swirski*

![Иллюстрация к проекту](https://github.com/chapych/Pupil-recogniser/blob/master/example1.jpg)
![Иллюстрация к проекту](https://github.com/chapych/Pupil-recogniser/blob/master/example2.jpg)
![Иллюстрация к проекту](https://github.com/chapych/Pupil-recogniser/blob/master/example3.jpg)

