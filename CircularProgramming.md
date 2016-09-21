###Circular Programming
Haskell 의 느긋한 계산법은 다음과 같은 직관적이지 않아보이는 코드를 작성할 수 있게 합니다. 아래 코드에서 변수 x 와 y 는 서로 상호 참조하고 있습니다.
```haskell
cyclic = let x = 0:y
             y = 1:x
         in x
```

    > take 5 cyclic
    [0,1,0,1,0,1,0,1,0,1]

위 코드가 전개되는 과정을 표현해보면 다음과 같습니다.

>   cyclic

> = x

> = 0:y

> = 0:1:x *`<-- 매듭! 다시 처음으로 되돌아갔습니다`*

> = 0:1:0:y

> = ...

```haskell
data Tree = Leaf | Fork Int Tree Tree deriving Show

t1 = Fork 9 (Fork 3 Leaf (Fork 1 (Fork 7 Leaf Leaf) (Fork 2 Leaf Leaf)))
            (Fork 11 Leaf (Fork 5 Leaf Leaf))

repMin:: Tree -> Tree
repMin = undefined
```
