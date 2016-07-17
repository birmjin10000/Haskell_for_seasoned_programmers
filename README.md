##베테랑 개발자들을 위한 Haskell programming

본 과정은 Haskell 기초를 충분히 알고 있고 Haskell 이외의 다른 프로그래밍 언어로 개발한 경험이 많은 개발자들을 대상으로 합니다. 이 과정을 마치면 다음 개념과 도구를 Haskell 프로그래밍에 사용하는데 불편함이 없게 되길 기대합니다.

Monad Transformers, Arrow, GADTs, Type Families, RankNTypes, Applicative Functor, QuickCheck, Parsec, ST monad, Zipper, Cabal, Haskell Tool Stack

학습 내용은 7시간 동안 다룰 수 있게 짜여져 있습니다.

## 사전 학습
- stackage
- cabal
- Sequence, Vector, Array
- ViewPatterns
- Pattern Synonyms

Haskell로 프로젝트를 할 때 cabal 을 통해 패키지를 설치하면 의존성 문제를 해결하는데 많은 시간이 쓰일 때가 있습니다. 이런 문제를 해결하고자 [stack](https://github.com/commercialhaskell/stack)과 같은 도구가 있습니다. Mac OS X의 경우 homebrew 로 설치하는 것이 가장 간편합니다. 설치 후에 다음과 같이 my-project 라는 이름으로 프로젝트를 하나 만들고 빌드 및 실행해 봅니다.

    stack new my-project new-template
    stack setup
    stack build
    stack exec my-project-exe

#### Sequence 자료형
Data.Map, Data.Set 모듈과 함께 cantainers 패키지에 있습니다. 구현이 finger tree로 되어 있는 매우 효율적인 자료구조입니다. List 를 사용하는 것이 너무 느린 경우에는 Sequence 를 쓰도록 합니다. Sequence 의 경우 맨 처음과 맨 끝을 접근하는 데 드는 시간과 Sequence 두 개를 붙이는 데 드는 시간이 모두 O(1) 이고, 중간에 있는 것을 접근하는데 드는 시간은 O(log n) 입니다. 단, Sequence 의 경우 유한자료구조입니다. 즉, List 와는 달리 strict & finite 합니다.
```haskell
import qualified Data.Sequence as Seq

Seq.empty -- fromList []
Seq.singleton 1 -- fromList [1]
a = Seq.fromList [1,2,3] -- fromList [1,2,3]
```
Sequence 에 사용할 수 있는 연산자들을 살펴봅시다.
```haskell
import Data.Sequence ((<|),(|>),(><))
import qualified Data.Sequence as Seq
import qualified Data.Foldable as F

a = 0 <| Seq.singleton 1 -- fromList [0,1], 맨 앞에 붙입니다, O(1)
b = Seq.singleton 1 |> 2 -- fromList [1,2], 맨 뒤에 붙입니다, O(1)
left = Seq.fromList [1,2,3]
right = Seq.fromList [7,8]
joined = left >< right -- fromList [1,2,3,7,8], 두 개를 이어 붙입니다, O(log(min(n1,n2)))
{- 아래의 index, update, take, drop, splitAt 함수는 모두 O(log(min(i,n-i))) 의 시간이 걸립니다. -}
thirdItem = Seq.index joined 2 -- 3, 색인으로 접근합니다.
modified = Seq.update 4 14 joined -- fromList [1,2,3,7,14], 특정 위치의 값을 바꿉니다.
take2 = Seq.take 2 joined -- fromList [1,2]
drop2 = Seq.drop 2 joined -- fromList [3,7,8]
splitted = Seq.splitAt 2 joined -- (fromList [1,2], fromList [3,7,8])
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
Pattern synonym 을 함께 쓰면 다음처럼 좀 더 편하게 할 수 있습니다.
```haskell
{-# LANGUAGE ViewPatterns, PatternSynonyms #-}
import qualified Data.Sequence as Seq

{-양방향 pattern synonyme 으로 Empty, (:<), (:>) 는 Sequence constructor 로도 사용할 수 있습니다 -}
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
vector 패키지에 있습니다. Vector 자료형은 색인이 정수인 배열입니다. 구현은 HAMT(Hash Array Mapped Trie)로 되어 있습니다. Immutable vector 와 Mutable vector, 둘 다 있습니다. Data.Vector 모듈이 Immutable vector 이고 Data.Vector.Mutable 모듈이 Mutable vector 입니다. Immutable vector의 경우 기본 자료형인 List와 동작특성이 완전히 같아서 1:1 로 대치할 수 있습니다. Fuctor 와 Foldable 같은 공통 type class 의 instance 이며 병렬코드에서도 잘 동작합니다. 물론 성능특성은 List 와는 확연히 다릅니다. 반면, Mutable vector 는 C 언어의 배열에 가깝습니다. Mutable vector 에 적용하는 연산들은 IO 또는 ST
Monad 를 통해 해야 합니다.

한편, vector 자료형은 담고 있는 자료의 형태에 따라 Boxed, Storable, Unboxed 의 세가지 구분이 또 있습니다.

* Boxed 에는 Haskell 의 어떤 종류의 자료도 담을 수 있습니다. 실제 자료는 heap memory 에 있고 vector 는 해당 자료에 대한 pointer 만 담고 있습니다. Pointer 사용에 따른 추가 메모리 사용이 있으며 pointer derefenecing 에 따른 성능손실이 있습니다.
* Storable 과 Unboxed 는 byte array 로 자료를 직접 담고 있습니다. 따라서 Boxed 에 비해 추가 메모리 사용이 없고, pointer dereferencing 으로 인한 성능손실도 없으며 cache 적중율도 우수합니다. Storable 과 Unboxed 의 차이는 조금 사소한데, 다음과 같습니다.
    * Storable 에 저장하는 자료는 Storable type class 에 속해야 하며 이는 malloc 으로 확보한 메모리에 담깁니다. malloc 으로 확보한 메모리영역은 pinned 됩니다. 즉, Garbage collector 가 이 메모리 영역을 옮길 수 없습니다. 따라서 메모리 파편화가 발생하지만 대신 C FFI(Foreign Function Interface)에게 해당 메모리 영역을 공유할 수 있습니다.
    * Unboxed 에 저장하는 자료는 Prim type class (이는 primitive 패키지에 있습니다)에 속해야 하며 이는 Garbage collector 가 관리하는 unpinned 메모리에 담깁니다. 따라서 Storable 과는 정반대의 특성을 가집니다.

각각의 Vector 자료형 구현에 해당하는 모듈은 다음과 같습니다.

<table>
  <tr><td rowspan="2">Boxed</td><td>Data.Vector</td><td>Immutable</td></tr>
  <tr><td>Data.Vector.Mutable</td><td>Mutable</td></tr>
  <tr><td rowspan="2">Unboxed</td><td>Data.Vector.Unboxed</td><td>Immutable</td></tr>
  <tr><td>Data.Vector.Unboxed.Mutable</td><td>Mutable</td></tr>
  <tr><td rowspan="2">Storable</td><td>Data.Vector.Storable</td><td>Immutable</td></tr>
  <tr><td>Data.Vector.Storable.Mutable</td><td>Mutable</td></tr>
</table>

그렇다면 언제 어떤 형태의 Vector 를 써야 할까요? 어떤 Vector 구현이 자신의 용도에 맞는지는 결국 profiling 과 benchmarking 으로 확인해야 합니다. 다만 일반적인 지침은 다음과 같습니다.

* Unboxed: 대부분의 용도에 이걸 씁니다.
* Boxed: 사용하려는 자료구조가 복합적인 경우에는 이걸 씁니다.
* Storable: C FFI 를 사용할 때는 이걸 씁니다.

Vector의 종류가 세 개나 되는데 그 때마다 서로 다른 API 를 써야 된다면 Vector 를 사용하는 라이브러리를 만들 때는 세 가지 인터페이스를 모두 작성해야 하는 불편함이 있을 것입니다. 그래서 **Data.Vector.Generic** 모듈과 **Data.Vector.Generic.Mutable** 모듈이 있습니다. Immutable vector 에 대한 인터페이스는 Data.Vector.Generic 에 정의되어 있고, Mutable vector 에 대한 인터페이스는 Data.Vector.Generic.Mutable 에 정의되어 있습니다. 라이브러리를 작성하는 경우에는 이 인터페이스를 사용합니다.

한편, 기본 자료형인 List와 Vector를 비교하면 다음과 같습니다.
Haskell 의 List 는 Immutable, Singly-linked list 입니다. 리스트의 맨 앞에 뭔가를 붙일 때마다 그 새로운 것을 위한 heap 메모리를 할당하고 원래 리스트의 맨 앞을 가리킬 포인터를 만들고, 새로 붙인 것을 가리킬 포인터를 만듭니다. 이렇게 포인터를 여러 개 가지고 있으니까 메모리도 많이 잡아먹고 리스트 순회나 색인접근 같은 동작은 시간이 오래 걸립니다(N 번째 항목을 가져오려면 N 번의 pointer dereferencing 이 필요함).
반면, Vector 는 하나의 메모리 영역을 통째로 할당하여 사용합니다. 그래서 임의 위치 접근에 필요한 시간이 항상 일정하고, 새로 항목을 추가할 때 메모리도 적게 듭니다. 하지만 맨 앞에 뭔가를 추가할 때는 List 와 비교할 때 매우 효율이 낮습니다. 왜냐하면 새로 연속된 메모리 영역을 할당한 다음 옛날 것들을 복사하고, 새로운 항목을 추가하는 식으로 동작하기 때문입니다.

이제 Vector 자료형의 사용법을 살펴보겠습니다. List 에서 사용하는 함수와 사용법과 동작이 완전히 같습니다.
```haskell
import qualified Data.Vector as V

v = V.enumFromN 1 10::V.Vector Int
V.filter odd v -- [1,3,5,7,9]
V.map (*2) v -- [2,4,...,20]
V.dropWhile (<6) v -- [6,7,8,9,10]
V.head v -- 1
V.foldr (+) 0 v -- 55
```
이번에는 Unboxed 형태로 사용해보겠습니다. 위의 Boxed 형태와 똑같은 인터페이스를 제공합니다.
```haskell
import qualified Data.Vector.Unboxed as U

v = U.enumFromN 1 10::U.Vector Int
U.filter odd v -- [1,3,5,7,9]
U.map (*2) v -- [2,4,...,20]
-- 이후 생략..
```
그리고 Vector 를 만들 때 replicate 와 replicateM 함수를 자주 사용합니다.
```haskell
import qualified Data.Vector.Unboxed as U
import System.Random (randomRIO)

myV::U.Vector Char
myV = U.replicate 10 'a' -- 길이 10 짜리 vector 를 만들고 값을 'a' 로 채웁니다.

myV2::IO (U.Vector Int)
myV2 = U.replicateM 20 $ do -- 여기서 replicateM 은 monadic action 을 20번 수행한 결과로 벡터를 만듭니다.
         i <- randomRIO(0,9)
         return i
```
위 코드를 ghci 에서 load 한 다음 Vector 의 내용들을 다음처럼 찍어볼 수 있습니다.

    > print myV
    "aaaaaaaaaa"
    > print =<< myV2
    [3,2,5,2,3,8,0,5,7,0,2,8,0,8,0,3,5,0,6,7]

Mutable vector 예시를 보겠습니다. 아래 코드는 0부터 9사이의 숫자로 난수를 10<sup>6</sup>개 만들었을때 얼마만큼의 빈도로 각 숫자가 나오는지 보여줍니다.
```haskell
import qualified Data.Vector.Unboxed.Mutable as UM
import Data.Vector.Unboxed (freeze)
import System.Random (randomRIO)

showDistribution:: IO ()
showDistribution = do
  v <- UM.replicate 10 (0::Int) -- 크기가 10인 vector 를 만들고 0으로 채웁니다.
  UM.replicateM (10^6) $ do -- do block(아래 세 줄) 을 10^6 번 실행합니다.
    i <- randomRIO (0,9)
    oldCount <- UM.read v i -- vector의 i 번째 위치의 값을 읽어옵니다.
    UM.write v i (oldCount + 1) -- mutable vector의 i 번째 위치의 값을 갱신합니다.
  immutableV <- freeze v -- Immutable 한 복사본을 만듭니다.
  print immutableV
```
이번에는 Boxed & Mutable vector 의 예시를 보겠습니다.
```haskell
import qualified Data.Vector.Mutable as M
import qualified Data.Vector as V(create,Vector())
data Point = Point Double Double
data PointSum = PointSum Int Double Double
addToPointSum::Point -> PointSum -> PointSum
addToPointSum (Point x y) (PointSum n xs ys) = PointSum (n+1) (x+xs) (y+ys)

assign:: [Point] -> V.Vector PointSum
assign points = V.create $ do -- create 함수는 do block 에 있는 monadic action 을 수행하여 Vector 를 만듭니다.
  vector <- M.replicate 2 (PointSum 0 0 0)
  let addPoint p@(Point x y) = do
      let index = if (y >= x) then 0 else 1
      pointSum <- M.read vector index
      M.write vector index $! addToPointSum p pointSum
  mapM_ addPoint points
  return vector
```
위 코드에서는 PointSum 자료형을 원소로 하는 Boxed vector 를 만들었습니다. 이처럼 Int, Char 와 같은 단순 자료형이 아닌 복합 자료형의 경우 Boxed vector 를 사용합니다. 만약에 PointSum 자료형을 가지고 Unboxed vector 를 만들려고 한다면 PointSum 자료형을 Unboxed instance 로 만들어야 하는데 이 경우 적지 않은 코드를 프로그래머가 작성해주어야 합니다. 예를 들어 다음 코드를 보면 Point3D 라는 자료형을 Unboxed 의 Instance 로 만들어주기 위해 작성해야 하는 코드양이 적지 않음을 알 수 있습니다.
```haskell
{-# LANGUAGE TypeFamilies, MultiParamTypeClasses #-}

import qualified Data.Vector.Generic as G
import qualified Data.Vector.Generic.Mutable as M
import Control.Monad (liftM, zipWithM_)
import Data.Vector.Unboxed.Base

data Point3D = Point3D Int Int Int

newtype instance MVector s Point3D = MV_Point3D (MVector s Int)
newtype instance Vector    Point3D = V_Point3D  (Vector    Int)
instance Unbox Point3D

instance M.MVector MVector Point3D where
  basicLength (MV_Point3D v) = M.basicLength v `div` 3
  basicUnsafeSlice a b (MV_Point3D v) = MV_Point3D $ M.basicUnsafeSlice (a*3) (b*3) v
  basicOverlaps (MV_Point3D v0) (MV_Point3D v1) = M.basicOverlaps v0 v1
  basicUnsafeNew n = liftM MV_Point3D (M.basicUnsafeNew (3*n))
  basicInitialize _ = return ()
  basicUnsafeRead (MV_Point3D v) n = do
              [a,b,c] <- mapM (M.basicUnsafeRead v) [3*n,3*n+1,3*n+2]
              return $ Point3D a b c
  basicUnsafeWrite (MV_Point3D v) n (Point3D a b c) =
              zipWithM_ (M.basicUnsafeWrite v) [3*n,3*n+1,3*n+2] [a,b,c]

instance G.Vector Vector Point3D where
  basicUnsafeFreeze (MV_Point3D v) = liftM V_Point3D (G.basicUnsafeFreeze v)
  basicUnsafeThaw (V_Point3D v) = liftM MV_Point3D (G.basicUnsafeThaw v)
  basicLength (V_Point3D v) = G.basicLength v `div` 3
  basicUnsafeSlice a b (V_Point3D v) = V_Point3D $ G.basicUnsafeSlice (a*3) (b*3) v
  basicUnsafeIndexM (V_Point3D v) n = do
              [a,b,c] <- mapM (G.basicUnsafeIndexM v) [3*n,3*n+1,3*n+2]
              return $ Point3D a b c
```
이러한 작업을 좀 더 편하게 도와주는 vector-th-unbox 패키지도 있습니다. 성능이 매우 중요한 경우에는 이처럼 수고를 무릅쓰고 Unboxed vector 를 이용할 수 있습니다.

한편, vector-algorithms 이라는 패키지가 있는데 Vector 에 대해 사용할 수 있는 알고리즘들을 가지고 있습니다. 주로 정렬에 관한 알고리즘들인데 다음 예제를 보겠습니다.
```haskell
import Data.Vector.Algorithms.Merge (sort)
import qualified Data.Vector.Generic.Mutable as M
import qualified Data.Vector.Unboxed as U
import System.Random (randomRIO)

mergeSortV = do
  v <- M.replicateM 100 $ randomRIO (0, 999::Int)
  sort v
  U.freeze v >>= print
```

#### Array 자료형
array 패키지에 있습니다. array 패키지의 경우 색인으로 정수뿐만 아니라 Ix("Indexable" 정도의 뜻으로 생각) type class 에 속하는 모든 것을 사용할 수 있다는 특징이 있습니다. array 패키지도 Immutable 과 Mutable 두 가지 구현을 제공하는데 기본이 되는 Data.Array 모듈은 Immutable & Lazy 합니다. 그래서 피보나치 수열 만들기, 동적계획법 등의 경우에 사용하기 좋습니다. 이러한 경우 말고는 연속된 구조의 자료가 필요할 때는 대부분의 경우 vector 패키지를 사용하도록 합니다. array 패키지보다 vector 패키지가 훨씬 성능도 우수하고 API도 풍부합니다. vector 패키지는 stream fusion 이라는 컴파일러 최적화 기법을 이용하도록 작성되어 있습니다. vector 패키지가 efficient array 를 표방하고 있음을 기억합시다.

Data.Array 모듈의 사용 예를 보겠습니다. array constructor 는 array, listArray, accumArray 이렇게 세 개가 있습니다.
먼저 array 함수는 두 개의 인자를 받습니다. 첫 번째 인자는 (1, 3) 과 같은 색인의 범위를 tuple 로 받습니다. 두 번째 인자는 (색인, 값) tuple 의 목록, 즉 association list 입니다. 다음 코드를 봅니다.
```haskell
import Data.Array

a = array(1,3)[(1,'a'),(2,'b'),(3,'c')] -- array (1,3) [(1,'a'),(2,'b'),(3,'c')]

a!1 -- 'a', 색인으로 값을 가져옵니다.
bounds a -- (1,3), 배열의 최소색인과 최대색인의 쌍을 가져옵니다.
indices a -- [1,2,3], 배열의 색인 목록을 가져옵니다.
elems a -- "abc", 배열의 값들의 목록을 가져옵니다.
assocs a -- [(1,'a'),(2,'b'),(3,'c')], 배열의 (색인, 값) 쌍 목록을 가져옵니다.
```
다음으로 listArray 함수는 array 함수와 달리 두번째 인자로 (색인,값) tuple 의 목록 대신 값의 목록을 받습니다.
```haskell
b = listArray (1,3) ['a'..] -- array (1,3) [(1,'a'),(2,'b'),(3,'c')]
```
마지막으로 accumArray 함수는 accumlated array 를 만든다는 뜻으로 첫번째 인자로 accumulating function 을 받습니다. 이는 마지막 인자로 받는 association list 에서 겹치는 색인이 있을 경우 이 색인들의 값을 accumulating function 으로 합치기 때문입니다. 다음 코드를 봅시다.
```haskell
c = accumArray (+) 1 ('a','c') [('a',1),('b',2),('a',3),('a',4)] -- array ('a','c') [('a',9),('b',3),('c',1)]

```
이제 Array 를 이용하여 fibonacci 수열을 만들어봅시다. Lazy evalution 을 이용합니다.
```haskell
fibonacci:: Int -> Array Int Integer
fibonacci n = a where a = array (0,n) ([(0,1),(1,1)] ++ [(i, a!(i-2) + a!(i-1))|i<-[2..n]])
```
만약 Array 의 일부 값을 바꾸고 싶을 때는 다음의 incremental update 함수를 이용합니다.
```haskell
c // [('a',11)] -- array ('a','c') [('a',11),('b',3),('c',1)]
```
#### 언제 뭘 쓸까요?
Data.Sequence, Data.Vector, Data.Array 는 모두 순차적인 자료구조입니다. 언제 뭘 써야 할까요? 먼저 Array 와 Vector 를 비교하면, 대부분의 경우 Vector를 쓰도록 합니다. 이유는 앞에서 설명했습니다. 그렇다면 Sequence 와 Vector 를 비교하면 어떨까요. Vector 는 하나로 이어진 메모리영역을 통째로 잡고 쓰는 만큼 메모리공간을 효율적으로 사용합니다. 그러나 두 개의 Vector 를 이어붙일 때, 그리고 복사할 때의 성능은 좋지 않습니다. 반면, Sequence 는 tree 를 자료구조로
사용하기 때문에 Vector 보다는 메모리를 많이 사용합니다. 하지만 두 개의 Sequence 를 이어붙이는 동작의 성능은 O(log(min(n1,n2))) 로 우수하고, 복사 역시 O(log n) 으로 성능이 준수합니다. 정리하면, 아주 많은 양의 자료를 순차적으로 또는 임의 접근으로 처리할 때는 Vector 를 사용하고, 자료를 이어 붙이거나 쪼개는 동작을 많이 해야 할 때는 Sequence 를 쓰도록 합니다.

                  |Sequence          | Vector
:----------------:|:----------------:|:----------------:
자료구조          |Finger tree       |HAMT
임의접근          |O(log n)          |O(1)
두개 합치기       |O(log(min(n1,n2)))|O(n1+n2)
앞뒤에 하나 붙이기|O(1)              |O(n)

숙제) 지뢰찾기 게임을 Haskell로 구현해 보세요. 다음 MineSweeper.hs 코드를 완성해서 제출하세요.

## 숙제 복기 시간

## 첫 1시간
다음의 ghc 컴파일러 확장을 배웁시다.
- BinaryLiterals
- OverloadedStrings
- LambdaCase
- BangPatterns
- FlexibleInstances, TypeSynonymInstances
- MultiParamTypeClasses
- FunctionalDependencies
- RecordWildCards
- ParallelListComp
- TransformListComp
- FlexibleContexts
- RecursiveDo
- NoMonomorphismRestriction
- DeriveFunctor, DeriveFoldable, DeriveTraversable
- DeriveGeneric
- DeriveAnyClass
- DeriveDataTypeable
- GeneralizedNewtypeDeriving

GHC 컴파일러 확장은 꽤 종류가 많은데 그 중에는 여러 사람들이 대체로 사용을 권장하지 않는 것도 있습니다. 여기에서 소개하는 확장들도 꼭 사용을 권장하는 확장들만 있는것은 아닙니다. 그러나 소스 코드를 볼 때 비교적 자주 볼 수 있는 것들이기에 소개합니다.

#####BinaryLiterals
0b 또는 0B를 앞에 붙일 경우 그 다음에 나오는 숫자는 이진수를 뜻합니다. 즉 아래 코드에서 0b1101 은 이진수 1101 를 뜻합니다.
```haskell
{-# LANGUAGE BinaryLiterals #-}
a = 0b1101 -- 13
```
#####OverloadedStrings
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

문자열에 대해 다형성을 지원하도록 하려면 OverloadedStrings 확장을 사용합니다. 이렇게 하면 ByteString 이나 Text 도 String 처럼 IsString type class 의 instance 가 됩니다.
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
#####LambdaCase
case .. of 구문은 LambdaCase 확장을 이용하면 좀 더 간결하게 작성할 수 있습니다.
```haskell
{-# LANGUAGE LambdaCase #-}
sayHello names = map (\case
                   "호돌이" -> "호랑아, 안녕!"
                   "둘리" -> "공룡아, 안녕!"
                   name -> name ++", 반가워요!") names
```
#####BangPatterns
Haskell 의 lazy evaluation 은 stack을 많이 사용하는 상황을 만들 수 있습니다. 이 때 사용할 수 있는 것이 Bang Patterns 입니다. 이를 사용하면 eager evaluation 을 하도록 만들 수 있습니다. 다음 코드처럼 변수 이름 앞에 느낌표를 붙이면 해당 변수는 thunk 에서 value 로 평가됩니다.
```haskell
{-# LANGUAGE BangPatterns #-}
import Data.List (foldl')

mean::[Double] -> Double
mean xs = s / l
  where (s,l) = foldl' step (0,0) xs
        step (!x,!y) a = (x+a,y+1)
```
#####FlexibleInstances, TypeSynonymInstances
Haskell 에서 type class 의 인스턴스를 만들 때는 그 형식이 "type 이름 + type variable 목록" 이어야 합니다. 그래서 다음 처럼 이를 벗어난 인스턴스를 만들면 컴파일 에러가 납니다.
```haskell
class Something a where
  doSomething:: a -> Integer
instance Something [Char] where
  doSomething x = 1
```
    Illegal instance declaration for ‘Something [Char]’
      (All instance types must be of the form (T a1 ... an)
       where a1 ... an are *distinct type variables*,
       and each type variable appears at most once in the instance head.
       Use FlexibleInstances if you want to disable this.)
    In the instance declaration for ‘Something [Char]’

이 때는 FlexibleInstances 확장을 사용하면 좀 더 유연하게 인스턴스를 만들 수 있습니다. 이번에는 tuple 을 Vector 의 인스턴스로 만들어봅시다.
```haskell
{-# LANGUAGE FlexibleInstances #-}
class Vector v where
  distance:: v -> v -> Double
instance Vector (Double, Double) where
  distance (a,b) (c,d) = sqrt $ (c-a)^2 + (d-b)^2

d = distance (1,2) (8.2::Double, 9.9::Double) -- 10.688779163215974
```
한편, FlexibleInstances 확장의 사용은 또 다른 확장인 TypeSynonymInstances 확장을 내포하고 있습니다. 다음 코드를 봅시다.
```haskell
type Point a = (a,a)
instance C (Point a) where
```
위 코드를 컴파일하면 다음과 같은 에러가 납니다. 
Error 가 나는 이유는...
```haskell
{-# LANGUAGE TypeSynonymInstances #-}

```
#####MultiParamTypeClasses
지금까지는 type class 를 만들 때 type variable 을 하나만 사용했습니다. 그런데 다음과 같은 경우에는 type parameter 가 두 개가 필요합니다. container 를 뜻하는 type class 를 만들려면 다음과 같이 할 수 있을 겁니다. 그런데 이를 컴파일하면 에러가 납니다.
```haskell
{-# LANGUAGE FlexibleInstances #-}
class Eq e => Collection c e where
  insert:: c -> e -> c
  member:: c -> e -> Bool

instance Eq a => Collection [a] a where
  insert xs x = x:xs
  member = flip elem
```
    Too many parameters for class ‘Collection’
    (Use MultiParamTypeClasses to allow multi-parameter classes)
    In the class declaration for ‘Collection’

이 때 MultiParamTypeClasses 확장을 이용하면 여러 개의 type variable 을 받을 수 있는 type class 를 정의할 수 있습니다. 직접 해보시기 바랍니다.

그런데 이렇게 정의했을 때 이 type class 정의에서 우리는 이미 알고 있지만 컴파일러는 모르는 정보가 생겼습니다. 그건 바로 Collection 의 type 이 해당 Collection 의 원소의 type 을 결정한다는 정보입니다. 무슨 말이냐하면 어떤 Collection 의 type 이 [a] 꼴이면 그것의 원소의 type 은 a 가 된다는 것입니다. 예를 하나 더 들어보면 Collection 의 type 이 Hashmap a 이면 그것의 원소의 type 은 a 가 되는 것이 자명합니다. 우리는 이 정보를 알고 있지만, 우리가 Collection type class 를 정의한 코드에서는 이것에 대한 정보가 없기 때문에 compiler 는 이에 대한 정보를 알지 못한 상황이 됩니다. 그 결과 Compiler 는 필요 이상으로 일반화된 type 의 함수를 추론하게 됩니다. 예를 들어 Collection 에 원소를 두 개 추가하는 다음과 같은 함수를 정의했다고 합시다.
```haskell
ins2 xs a b = insert (insert xs a) b
```
이 함수의 type 을 컴파일러가 어떻게 추론했는지 확인해보면 다음과 같은 꼴로 추론함을 확인할 수 있습니다.

    > :t ins2
    ins2::(Collection c e1, Collection c e) => c -> e1 -> e -> c

이는 우리가 원하는 결과가 아닙니다. e1 과 e 가 같은 type 이라는 것을 compiler 가 모르기 때문에 이처럼 지나치게 일반화된 type 으로 추론을 했습니다. 이 같은 문제를 해결할 수 있는 것이 다음의 Functional Dependency 확장입니다.
#####FunctionalDependencies
아래 코드처럼 Functional Dependency 확장을 이용하면 ins2 함수의 type 을 컴파일러가 어떻게 추론하는지 봅시다.
```haskell
{-# LANGUAGE FunctionalDependencies #-}
class Eq e => Collection c e | c -> e where
  insert:: c -> e -> c
  member:: c -> e -> Bool
```
위 코드에서 수직선 뒷 부분의 c -> e 가 뜻하는 바는 c 가 e 의 type 을 결정한다는 뜻입니다. 이제 ins2 의 type 을 확인해 보면 다음처럼 원하는 결과를 얻을 수 있습니다.

    > :t ins2
    ins2::Collection c e => c -> e -> e -> c


#####RecordWildCards
RecordWildCards 확장의 주 목적은 코드를 좀 더 간결하게 보이도록 하는 것입니다. 다음과 같은 Record syntax 의 자료형이 있다고 합시다.
```haskell
data Worker = Worker
   { workerName :: String
   , workerPosition :: String
   , workerFirstYear :: Int }
```
이를 Data.Aeson 모듈을 이용해서 JSON 형식으로 바꾸려고 합니다. 그러려면 Worker 자료형이 Data.Aeson 모듈의 FromJSON 과 ToJSON 의 instance 이어야 합니다. 그래서 다음처럼 코드를 작성합니다. ToJSON 의 인스턴스로 만드는 코드 예만 들겠습니다.
```haskell
instance ToJSON Worker where
  toJSON w = object [ "name" .= workerName w
                    , "position" .= workerPosition w
                    , "first-year" .= workerFirstYear w ]
```
그런데 이 코드를 보면 w 변수가 군더더기처럼 모든 필드에 나오고 있습니다. 이러한 때에 RecordWildCards 확장을 쓰면 다음처럼 좀 더 깔끔하게 코드를 작성할 수 있습니다.
```haskell
{-# LANGUAGE RecordWildCards #-}
instance ToJSON Worker where
  toJSON Worker{..} = object [ "name" .= workerName
                             , "position" .= workerPosition
                             , "first-year" .= workerFirstYear ]
```
위 코드에서 Worker{..} 부분이 RecordWildCards 확장을 씀으로 인해 가능한 코드로서 Worker constructor 에 대한 pattern match 입니다. 이 부분에서 Worker 자료형의 모든 필드에 대한 binding이 이루어집니다. 이 예에서는 필드 갯수가 몇 개 되지 않아서 큰 차이는 없지만 많은 필드를 가진 자료형의 경우는 차이가 벌어집니다. {..} 를 이용한 pattern match 는 Constructor 전체가 아닌 일부 필드에 대해서만도 할 수 있습니다. 다음 코드를 봅시다.
```haskell
{-# LANGUAGE RecordWildCards #-}
data C = C {a :: Int, b :: Int, c :: Int, d :: Int}
f (C {a = 1, ..}) = b + c + d
f (C {..}) = b * c * d
```
이렇게 코드를 작성했을때 f (C 1 2 3 4) 의 결과는 9 가 되고 f (C 9 2 3 4) 의 결과는 24가 됩니다.
#####ParallelListComp
List comprehension 에서는 Cartesian product 가 나옵니다. 즉, [x+y|x<-[1..3],y<-[10..12]] 의 결과는 길이가 9인 List 가 됩니다. ParallelListComp 확장을 쓰면 각 원소들을 1:1 대응하여 연산을 수행합니다. ParallelListComp 확장의 경우 generator 간 구분은 쉼표가 아니라 수직선으로 합니다.
```haskell
{-# LANGUAGE ParallelListComp #-}
[x+y|x<-[1..3] | y <-[10..12]] -- 결과는 [11,13,15]
```
이는 zipWith (+) [1..3] [10..12] 한 것과 같은 결과로서 ParallelListComp 를 이용한 표현식은 zipWith 를 이용하여 똑같이 작성할 수 있습니다. 그럼에도 ParallelListComp 확장을 쓰면 좋은 점은 코드를 좀 더 보기좋게 작성할 수 있다는 점에 있습니다.
#####TransformListComp
TransformListComp 는 List Comprehension 의 기능을 더욱 확장한 것으로 볼 수 있는데 이 확장을 사용하면 마치 SQL query 를 작성하듯 grouping, sorting 기능들을 써서 List comprehension 을 작성할 수 있습니다.
```haskell
{-# LANGUAGE TransformListComp #-}
import GHC.Exts (sortWith, groupWith, the)
a = [x*y| x<-[-1,1,-2], y<-[1,2,3], then reverse] -- [-6,-4,-2,3,2,1,-3,-2,-1]
b = [x*y| x<-[-1,1,-2], y<-[1,2,3], then sortWith by x] -- [-2,-4,-6,-1,-2,-3,1,2,3]
c = [(the p, m)| x<-[-1,1,-2], y<-[1,2,3], let m = x*y, let p = m > 0
                                         , then group by p using groupWith]
-- [(False,[-1,-2,-3,-2,-4,-6]),(True,[1,2,3])]
```
이 코드에서 TransformListComp 확장을 씀으로 인해 사용할 수 있는 부분은 then f, then f by e, then group by e using f 구문입니다. 이처럼 TransformListComp 에서 사용할 수 있는 qualifier 구문의 뜻을 살펴봅시다.
- *then* f 는 말 그대로 List 가 만들어진 후에 List에 함수 f 를 적용하여 새로운 List 를 만듭니다. 함수 f 는 [a] -> [a] 꼴입니다.
```haskell
{-# LANGUAGE TransformListComp #-}
[x*y|x<-[1,2,3], y<-[6,7], then take 3] -- [6,7,12]
```
- *then* f *by* e 는 위와 비슷한데 함수 f(함수 f 는 (a -> b)->[a]->[a] 꼴 입니다) 의 첫번째 인자로 쓰일 함수는 compiler 가 만들어서 전달합니다.
```haskell
{-# LANGUAGE TransformListComp #-}
import Data.List (sortOn)
import GHC.Exts (sortWith)

sortWith (>5) [1,9,5,7,8,2] -- [1,5,2,9,7,8]
[(x*y,y)|x<-[1,2],y<-[7,6,8], then sortWith by y] -- [(6,6),(12,6),(7,7),(14,7),(8,8),(16,8)]
sortOn snd [(x*y,y)|x<-[1,2],y<-[7,6,8]] -- sortOn 함수를 써서 같은 결과를 얻을 수 있습니다.
```
- *then group by* e *using* f 는 함수 f 로 List comprehension 결과를 끼리끼리 묶습는다. 이 함수 f 는 이전과 마찬가지로 (a -> b)->[a]->[a] 꼴이며 이것의 첫번째 인자는 컴파일러가 값 e 를 이용하여 만들어서 함수 f에 전달합니다.
```haskell
{-# LANGUAGE TransformListComp #-}
import GHC.Exts (groupWith, the)

the [1,1,1] -- 1
groupWith (`mod` 3) [1,9,8,3,6,5,4,7] -- [[9,3,6],[1,4,7],[8,5]]
[(the p, m)| x<-[-1,1,-2], y<-[1,2,3], let m = x*y, let p = m `mod` 3, then group by p using groupWith]
-- [(0,[-3,3,-6]),(1,[-2,1,-2]),(2,[-1,2,-4])]
```
- *then group using* f 에서 함수 f 는 [a] -> [[a]] 꼴입니다.
```haskell
{-# LANGUAGE TransformListComp #-}
[y|x<-[1..3], y<-"cat", then group using inits]
-- ["","c","ca","cat","catc","catca","catcat","catcatc","catcatca","catcatcat"]
inits [y|x<-[1..3], y<-"cat"] -- 같은 결과를 얻습니다.
[(x,y)|x<-[1,2], y<-"hi", then group using inits]
-- [([],""),([1],"h"),([1,1],"hi"),([1,1,2],"hih"),([1,1,2,2],"hihi")]
map (foldr (\(num,ch) acc -> (num:fst acc, ch:snd acc)) ([],[])) $ inits [(x,y)|x<-[1,2], y<-"hi"] -- 같은 결과
```
#####FlexibleContexts

#####RecursiveDo

#####NoMonomorphismRestriction

#####DeriveFunctor, DeriveFoldable, DeriveTraversable
다음처럼 Tree 를 정의하고 이에 대해서 fmap 함수를 적용하려면 Tree 가 Fuctor 이어야 합니다. 즉, 직접 Tree 를 Functor 로 만들어주어야 하는데, 이 때 DeriveFunctor 확장을 쓰면 컴파일러가 이 작업을 대신 해 줍니다. 마찬가지로 fold 함수를 적용하려면 Tree 가 Foldable 이어야 하는데 이 때도 역시 DeriveFoldable 확장을 쓰면 컴파일러가 알아서 Tree 를 Foldable 로 만들어 줍니다. DeriveTraversable 도 마찬가지로 함수 traverse 를 적용할 수 있도록 해 줍니다.
```haskell
{-# LANGUAGE DeriveFunctor, DeriveFoldable, DeriveTraversable #-}
data Tree a = Empty | Fork (Tree a) a (Tree a) deriving (Show, Functor, Foldable, Traversable)
myT = Fork (Fork Empty "Daenerys" Empty) "Jon" (Fork Empty "Arya" (Fork Empty "Sansa" Empty))
fmap length myT -- Fork (Fork Empty 8 Empty) 3 (Fork Empty 4 (Fork Empty 5 Empty))
foldr (:) [] myT -- ["Daenerys","Jon","Arya","Sansa"]
traverse (\name -> putStrLn ("What's "++name++"'s occupation?") *> getLine) myT
-- What's Daenerys's occupation?
-- ...
```

#####DeriveGeneric

#####DeriveAnyClass

#####DeriveDataTypeable

#####GeneralizedNewtypeDeriving

## 두 번째 시간
다음의 ghc 컴파일러 확장을 배웁시다.
- RankNTypes
- GADTs(Generalised Algebraic Data Types)
- ScopedTypeVariables
- LiberalTypeSynonyms
- ExistentialQuantification
- TypeFamillies
- DefaultSignatures
- ConstraintKinds
- DataKinds
- PolyKinds
- KindSignatures

## 세 번째 시간
- Standalone deriving
- Typed holes
- REPA(REgular PArallel arrays)

## 네 번째 시간
- DWARF based debugging
- Template Haskell with Quasiquoting

## 다섯 번째 시간
- Dependent Types

## 여섯 번째 시간

## 더 읽을 거리
#### Zipper
#### Finger trees
#### Hash Array Mapped Trie (HAMT)
#### Stream fusion

## License
Eclipse Public License
