# Optimization Methods Lab 1

## Файлы

- [common/oracle](common/oracle.py) &ndash; обертка над функцией и ее градиентом
- [gradient](gradient.py) &ndash; функция градиентного спуска
- [optimize](optimize) &ndash; одномерные линейные оптимизаторы
  - [optimizer](optimize/optimizer.py) &ndash; базовый класс
  - [methods](optimize/unidimensional) &ndash; методы [дихотомии](optimize/unidimensional/bisection.py), 
  [золотого сечения](optimize/unidimensional/golden_ratio.py) и [Фибоначчи](optimize/unidimensional/fibonacci.py)
