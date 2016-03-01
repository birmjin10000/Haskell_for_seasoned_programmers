##베테랑 개발자들을 위한 Haskell programming

본 과정은 Haskell 기초를 충분히 알고 있고 Haskell 이외의 다른 프로그래밍 언어로 개발한 경험이 많은 개발자들을 대상으로 합니다. 이 과정을 마치면 다음 개념과 도구를 Haskell 프로그래밍에 사용하는데 불편함이 없게 되길 기대합니다.

Monad Transformers, Arrow, GADTs, Type Families, RankNTypes, Applicative Functor, QuickCheck, Parsec, ST monad, Cabal

학습 내용은 7시간 동안 다룰 수 있게 짜여져 있습니다.

## 사전 학습
- stackage
- cabal
- Vector, Array, Sequence

Haskell로 프로젝트를 할 때 cabal 을 통해 패키지를 설치하면 의존성 문제를 해결하는데 많은 시간이 쓰일 때가 있습니다. 이런 문제를 해결하고자 [stack](https://github.com/commercialhaskell/stack)과 같은 도구가 있습니다. Mac OS X의 경우 homebrew 로 설치하는 것이 가장 간편합니다. 설치 후에 다음과 같이 my-project 라는 이름으로 프로젝트를 하나 만들고 빌드 및 실행해 봅니다.

    stack new my-project new-template
    stack setup
    stack build
    stack exec my-project-exe

#### Sequence 자료형
Data.Map, Data.Set 모듈과 함께 cantainers 패키지에 있습니다. 구현이 finger tree로 되어 있는 매우 효율적인 자료구조입니다. List 를 사용하는 것이 너무 느린 경우에는 Sequence 를 쓰도록 합니다. Sequence 의 경우 맨 처음과 맨 끝을 접근하는 데 드는 시간과 Sequence 두 개를 붙이는 데 드는 시간이 모두은 O(1) 이고, 중간에 있는 것을 접근하는데 드는 시간은 O(log n) 입니다. 단, Sequence 의 경우 유한자료구조입니다.
```haskell
import qualified Data.Sequence as Seq

Seq.empty -- fromList []
Seq.singleton 1 -- fromList []
a = Seq.fromList [1,2,3] -- fromList [1,2,3]
```
Sequence 에 사용할 수 있는 연산자들을 살펴봅시다.
```haskell
import Data.Sequence ((<|),(|>),(><))
import qualified Data.Sequence as Seq
import qualified Data.Foldable as F

a = 0 <| Seq.singleton 1 -- fromList [0,1], 맨 앞에 붙입니다.
b = Seq.singleton 1 |> 2 -- fromList [1,2], 맨 뒤에 붙입니다.
left = Seq.fromList [1,2,3]
right = Seq.fromList [7,8]
joined = left >< right -- fromList [1,2,3,7,8], 두 개를 이어 붙입니다.
thirdItem = Seq.index joined 2 -- 3, 색인으로 접근합니다.
take2 = Seq.take 2 joined -- fromList [1,2]
drop2 = Seq.drop 2 joined -- fromList [3,7,8], take과 drop 함수는 O(log(min(i,n-i))) 시간이 걸립니다.
totalSum = F.foldr1 (+) joined -- 21
listForm = F.toList joined -- [1,2,3,7,8]
```
Sequence 자료형의 Pattern matching 은 다음처럼 ViewPatterns ghc 확장을 이용합니다.
```haskell
{-# LANGUAGE ViewPatterns #-}
import qualified Data.Sequence as Seq

sumSeq:: Seq.Seq Int -> Int
sumSeq (Seq.viewl -> Seq.EmptyL) = 0
sumSeq (Seq.viewl -> (x Seq.:< xs)) = x + sumSeq xs
sum10 = sumSeq $ Seq.fromList [1..10] -- 55
```
Pattern sysnonym 을 함께 이용하면 다음처럼 좀 더 편하게 할 수 있습니다.
```haskell
{-# LANGUAGE ViewPatterns, PatternSynonyms #-}
import qualified Data.Sequence as Seq

pattern Empty <- (Seq.viewl -> Seq.EmptyL) where Empty = Seq.empty
pattern x:<xs <- (Seq.viewl -> (x Seq.:< xs)) where (:<) = (Seq.<|)
pattern xs:>x <- (Seq.viewr -> (xs Seq.:> x)) where (:>) = (Seq.|>)

sumSeq:: Seq.Seq Int -> Int
sumSeq Empty = 0
sumSeq (x:<xs) = x + sumSeq xs
sumTen = sumSeq $ Seq.fromList [1..10] -- 55
sum123 = sumSeq $ 1:<Seq.fromList [2,3] -- 6
```

#### Vector 자료형
vector 패키지에 있다. 

#### Array 자료형



숙제) 지뢰찾기 게임을 Haskell로 구현해 보세요. 다음 MineSweeper.hs 코드를 완성해서 제출하세요.

## 숙제 복기 시간

## 첫 1시간
다음의 ghc 컴파일러 확장을 배웁시다.
- BinaryLiterals
- OverloadedStrings
- LambdaCase
- ViewPatterns
- BangPatterns
- FlexibleInstances
- MultiParamTypeClasses
- TypeSynonymInstances
- ParallelListComp
- TransformListComp
- FunctionalDependencies
- FlexibleContexts
- RecordWildCards
- RecursiveDo
- NoMonomorphismRestriction
- DeriveFunctor, DeriveFoldable, DeriveTraversable
- DeriveGeneric
- DeriveAnyClass
- DeriveDataTypeable
- GeneralizedNewtypeDeriving

먼저 BinaryLiterals 확장은 0b 또는 0B를 앞에 붙일 경우 그 다음에 나오는 숫자는 이진수를 뜻합니다. 즉 아래 코드에서 0b1101 은 이진수 1101 를 뜻합니다.
```haskell
{-# LANGUAGE BinaryLiterals #-}
a = 0b1101 -- 13
```
다음 코드를 봅시다. 숫자의 type은 Int, Float, Double 등 여러가지인데, Haskell에서는 같은 숫자라도 주어진 type에 따라 type이 달리 정해질 수 있습니다. 숫자에 대해서는 다형성을 기본 지원해 주는 것이지요.

    > let a::Int; a = 2
    > let b::Double; b = 2
    > let c::Rational; c = 2

문자열도 String이 있고 유니코드를 위한 Text가 있는 등 여러 type 이 있습니다. Haskell에서는 문자열을 직접 변수에 바인딩하면 이것의 type은 항상 String이기 때문에 다음처럼 하면 동작하지 않습니다.

    > import Data.Text
    > let s::Text; s = "백두산"
    Couldn't match expected type ‘Text’ with actual type ‘[Char]’
    In the expression: "\48177\46160\49328"
    In an equation for ‘s’: s = "\48177\46160\49328"

문자열에 대해 다형성을 지원하도록 하려면 OverloadedStrings 확장을 사용합니다.
```haskell
{-# LANGUAGE OverloadedStrings #-}
import Data.Text
import Data.ByteString

a::Text
a = "백두산"
b::ByteString
b = "백두산"
c::String
c = "백두산"
```

case .. of 구문은 LambdaCase 확장을 이용하면 좀 더 간결하게 작성할 수 있습니다.
```haskell
{-# LANGUAGE LambdaCase #-}
sayHello names = map (\case
                   "호돌이" -> "호랑아, 안녕!"
                   "둘리" -> "공룡아, 안녕!"
                   name -> name ++", 반가워요!") names
```



## 두 번째 시간
다음의 ghc 컴파일러 확장을 배웁시다.
- RankNTypes
- GADTs(Generalised Algebraic Data Types)
- ScopedTypeVariables
- LiberalTypeSynonyms
- ExistentialQuantification
- TypeFamillies

## 세 번째 시간
- Pattern Synonyms
- Standalone deriving
- Typed holes

## 네 번째 시간
- DWARF based debugging
- Template Haskell with Quasiquoting

## 다섯 번째 시간
- Dependent Types

## 여섯 번째 시간

## 더 읽을 거리
#### Finger trees


## License
Eclipse Public License
