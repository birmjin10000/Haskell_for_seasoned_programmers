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
sumTen = sumSeq $ Seq.fromList [1..10] -- 55
```
Pattern sysnonym 을 함께 이용하면 다음처럼 좀 더 편하게 할 수 있습니다.
```haskell
{-# LANGUAGE ViewPatterns, PatternSynonyms #-}
import qualified Data.Sequence as Seq

{-양방향 pattern synonyme으로 Empty, (:<), (:>) 는 Sequence constructor 로도 사용할 수 있습니다 -}
pattern Empty <- (Seq.viewl -> Seq.EmptyL) where Empty = Seq.empty
pattern x:<xs <- (Seq.viewl -> (x Seq.:< xs)) where (:<) = (Seq.<|)
pattern xs:>x <- (Seq.viewr -> (xs Seq.:> x)) where (:>) = (Seq.|>)

sumSeq:: Seq.Seq Int -> Int
sumSeq Empty = 0
sumSeq (x:<xs) = x + sumSeq xs
sumTen = sumSeq $ Seq.fromList [1..10] -- 55
sum123 = sumSeq $ 1:<Seq.fromList [2,3] -- 6, (:<) 을 Sequence constructor 로 쓰고 있습니다
```

#### Vector 자료형
vector 패키지에 있습니다. Vector 자료형은 색인이 정수인 배열입니다. 구현은 HAMT(Hash Array Mapped Trie)로 되어 있습니다. Immutable vector 와 Mutable vector, 둘 다 있습니다. Data.Vector 모듈이 Immutable vector 이고 Data.Vector.Mutable 모듈이 Mutable vector 입니다. Immutable vector의 경우 기본 자료형인 List와 동작특성이 완전히 같습니다. 물론 성능특성은 다릅니다. 반면, Mutable vector 는 C 언어의 배열에 가깝습니다.

한편, vector 자료형은 담고 있는 자료의 형태에 따라 Boxed, Storable, Unboxed 의 세가지 구분이 또 있습니다.

* Boxed 에는 Haskell 의 어떤 종류의 자료도 담을 수 있습니다. 실제 자료는 heap memory 에 있고 vector 는 해당 자료에 대한 pointer 만 담고 있습니다. Pointer 사용에 따른 추가 메모리 사용이 있으며 pointer derefenecing 에 따른 성능손실이 있습니다.
* Storable 과 Unboxed 는 byte array 로 자료를 직접 담고 있습니다. 따라서 Boxed 에 비해 추가 메모리 사용이 없고, pointer dereferencing 으로 인한 성능손실도 없으며 cache 적중율도 우수합니다. Storable 과 Unboxed 의 차이는 조금 사소한데, 다음과 같습니다.
    * Storable 에 저장하는 자료는 Storable type class 에 속해야 하며 이는 malloc 으로 확보한 메모리에 담깁니다. malloc 으로 확보한 메모리영역은 pinned 됩니다. 즉, Garbage collector 가 이 메모리 영역을 옮길 수 없습니다. 따라서 메모리 파편화가 발생하지만 대신 C FFI(Foreign Function Interface)에게 해당 메모리 영역을 공유할 수 있습니다.
    * Unboxed 에 저장하는 자료는 Prim type class (이는 primitive 패키지에 있습니다)에 속해야 하며 이는 Garbage collector 가 관리하는 unpinned 메모리에 담깁니다. 따라서 Storable 과는 정반대의 특성을 가집니다.

각각의 Vector 자료형 구현에 해당하는 모듈은 다음과 같습니다.

* Data.Vector - Boxed, Immutable
* Data.Vector.Mutable - Boxed, Mutable
* Data.Vector.Storable - Storable, Immutable
* Data.Vector.Storable.Mutable - Storable, Mutable
* Data.Vector.Unboxed - Unboxed, Immutable
* Data.Vector.Unboxed.Mutable - Unboxed, Mutable

그렇다면 언제 어떤 형태의 Vector 를 써야 할까요? 어떤 Vector 구현이 자신의 용도에 맞는지는 결국 profiling 과 benchmarking 으로 확인해야 합니다. 다만 일반적인 지침은 다음과 같습니다.
담아둘 값이 Storable type class 의 instance 이면 Storable 을 씁니다. C FFI 가 필요없고 담아둘 값이 Prim type class의 instance이면 Unboxed 를 씁니다. 그 외의 모든 경우에는 Boxed 를 씁니다.

List와 Vector를 비교하면 다음과 같습니다.
Haskell 의 List 는 Immutable, Singly-linked list 입니다. 리스트의 맨 앞에 뭔가를 붙일 때마다 그 새로운 것을 위한 heap 메모리를 할당하고 원래 리스트의 맨 앞을 가리킬 포인터를 만들고, 새로 붙인 것을 가리킬 포인터를 만듭니다. 이렇게 포인터를 여러 개 가지고 있으니까 메모리도 많이 잡아먹고 리스트 순회나 색인접근 같은 동작은 시간이 오래 걸립니다.
반면, Vector 는 하나의 메모리 영역을 통째로 할당하여 사용합니다. 그래서 임의 접근에 필요한 시간이 항상 일정하고, 새로 항목을 추가할 때 메모리도 적게 듭니다. 하지만 맨 앞에 뭔가를 추가할 때는 List 와 비교할 때 매우 효율이 낮습니다. 왜냐하면 새로 연속된 메모리 영역을 할당한 다음 옛날 것들을 복사하고, 새로운 항목을 추가하는 식으로 동작하기 때문입니다.

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
#### Hash Array Mapped Trie (HAMT)


## License
Eclipse Public License
