###Circular Programming
Haskell 의 느긋한 계산법은 다음과 같은 직관적이지 않은 코드를 작성할 수 있게 합니다. 아래 코드에서 변수 x 와 y 는 서로 상호 참조하고 있습니다.
```haskell
cyclic = let x = 0:y
             y = 1:x
         in x
```

    > take 5 cyclic
    [0,1,0,1,0,1,0,1,0,1]

위 코드가 전개되는 과정을 표현해보면 다음과 같습니다. 보면 변수 x 와 y 각각의 한 쪽 끝을 다른 변수로 연결하는 것처럼 전개가 됩니다. 이러한 모습에서 이런 재귀적인 코드를 "Tying the Knot" 라는 말로 표현하기도 합니다.

cyclic = x = 0:y = 0:1:x *`(<-- 매듭! 다시 처음으로 되돌아갔습니다)`* = 0:1:0:y = ...

다른 예를 보겠습니다. Repmin 문제라고 하는데, Tree 자료구조에서 노드의 모든 값을 해당 Tree 의 가장 작은 값으로 바꾸는 문제입니다.

```haskell
data Tree = Leaf | Fork Int Tree Tree deriving Show

t1 = Fork 9 (Fork 3 Leaf (Fork 1 (Fork 7 Leaf Leaf) (Fork 2 Leaf Leaf)))
            (Fork 8 Leaf (Fork 5 Leaf Leaf))

repMin:: Tree -> Tree
repMin = undefined
```
위처럼 t1 이라는 Tree 가 있을 때 t1 의 모든 값을 가장 작은 값인 1 로 바꾸는 것입니다. 아래 그림처럼.

<img src="repmin.png">

