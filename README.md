##베테랑 개발자들을 위한 Haskell programming

본 과정은 Haskell 기초를 충분히 알고 있고 Haskell 이외의 다른 프로그래밍 언어로 개발한 경험이 많은 개발자들을 대상으로 합니다. 이 과정을 마치면 다음 개념과 도구를 Haskell 프로그래밍에 사용하는데 불편함이 없게 되길 기대합니다.

Monad Transformers, Arrow, GADTs, Type Families, RankNTypes, Applicative Functor, QuickCheck, Parsec, ST monad, Zipper, Cabal, Haskell Tool Stack

학습 내용은 이틀 동안 다룰 수 있게 짜여져 있습니다.

## 사전 학습
- stackage
- cabal
- Sequence, Vector, Array
- ViewPatterns
- Pattern Synonyms

Haskell로 프로젝트를 할 때 cabal 을 통해 패키지를 설치하면 의존성 문제를 해결하는데 많은 시간을 허비할 수 있습니다. 이런 문제를 해결하고자 [stack](https://github.com/commercialhaskell/stack)과 같은 도구가 있습니다. Mac OS X의 경우 homebrew 로 설치하는 것이 가장 간편합니다. 설치 후에 다음과 같이 my-project 라는 이름으로 프로젝트를 하나 만들고 빌드 및 실행해 봅니다.

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
위 코드를 GHCi 에서 load 한 다음 Vector 의 내용들을 다음처럼 찍어볼 수 있습니다.

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

숙제) 지뢰찾기 게임을 Haskell 로 구현해 보세요. 다음 MineSweeper.hs 코드를 완성해서 제출하세요.

## 숙제 복기 30분

## 첫날 첫 100분
다음의 ghc 컴파일러 확장을 배웁시다.
- [BinaryLiterals](#binaryliterals)
- [OverloadedStrings](#overloadedstrings)
- [LambdaCase](#lambdacase)
- [BangPatterns](#bangpatterns)
- [TupleSections](#tuplesections)
- [FlexibleInstances, TypeSynonymInstances](#flexibleinstances-typesynonyminstances)
- [MultiParamTypeClasses](#multiparamtypeclasses)
- [FunctionalDependencies](#functionaldependencies)
- [RecordWildCards](#recordwildcards)
- [ParallelListComp](#parallellistcomp)
- [TransformListComp](#transformlistcomp)
- [FlexibleContexts](#flexiblecontexts)
- [RecursiveDo](#recursivedo)
- [NoMonomorphismRestriction](#nomonomorphismrestriction)
- [DeriveFunctor, DeriveFoldable, DeriveTraversable](#derivefunctor-derivefoldable-derivetraversable)
- [DeriveGeneric, DeriveAnyClass](#derivegeneric-deriveanyclass)
- [DeriveDataTypeable](#derivedatatypeable)
    * [Data.Typeable](#datatypeable)
- [GeneralizedNewtypeDeriving](#generalizednewtypederiving)

GHC 컴파일러 확장은 꽤 종류가 많은데 그 중에는 여러 사람들이 대체로 사용을 권장하지 않는 것도 있습니다. 여기에서 소개하는 확장들도 꼭 사용을 권장하는 확장들만 있는것은 아닙니다. 그러나 소스 코드를 볼 때 비교적 자주 볼 수 있는 것들이기에 소개합니다.

####BinaryLiterals
0b 또는 0B를 앞에 붙일 경우 그 다음에 나오는 숫자는 이진수를 뜻합니다. 즉 아래 코드에서 0b1101 은 이진수 1101 를 뜻합니다.
```haskell
{-# LANGUAGE BinaryLiterals #-}
a = 0b1101 -- 13
```
####OverloadedStrings
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
####LambdaCase
case .. of 구문은 LambdaCase 확장을 이용하면 좀 더 간결하게 작성할 수 있습니다.
```haskell
{-# LANGUAGE LambdaCase #-}
sayHello names = map (\case
                   "호돌이" -> "호랑아, 안녕!"
                   "둘리" -> "공룡아, 안녕!"
                   name -> name ++", 반가워요!") names
```
####BangPatterns
Haskell 의 lazy evaluation 은 stack을 많이 사용하는 상황을 만들 수 있습니다. 이 때 사용할 수 있는 것이 Bang Patterns 입니다. 이를 사용하면 eager evaluation 을 하도록 만들 수 있습니다. 다음 코드처럼 변수 이름 앞에 느낌표를 붙이면 해당 변수는 thunk 에서 value 로 평가됩니다.
```haskell
{-# LANGUAGE BangPatterns #-}
import Data.List (foldl')

mean::[Double] -> Double
mean xs = s / l
  where (s,l) = foldl' step (0,0) xs
        step (!x,!y) a = (x+a,y+1)
```

####TupleSections
Tuple 을 만들 때 일부 요소를 partially applied 한 꼴을 이용할 수 있게 합니다. 다음 코드를 봅시다.
```haskell
{-# LANGUAGE TupleSections #-}
(1,) 2 -- (1,2)
(,3,) 1 5 -- (1,3,5)
map ("yo!",) [1,2,3] -- [("yo!",1),("yo!",2),("yo!",3)]
```

####FlexibleInstances, TypeSynonymInstances
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
####MultiParamTypeClasses
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

한편, MultiParamTypeClasses 확장은 반대로 type variable 이 전혀 없는 type class 도 정의할 수 있게 합니다. 다음 코드를 보면 Logger type class 는 type variable 이 전혀 없습니다. 이렇게 함으로써 사용자가 Logger 의 instance 를 단 하나만 만들 수 있게 할 수 있습니다.
```haskell
{-# LANGUAGE MultiParamTypeClasses #-}
class Logger where
  logMessage :: String -> IO ()

type Present = String
queueChristmasPresents :: Logger => [Present] -> IO ()
queueChristmasPresents presents = do
  mapM (logMessage . ("Queueing present for delivery: " ++)) presents
  return ()

instance Logger where
  logMessage t = putStrLn ("[XMAS LOG]: " ++ t)
```

여러 개의 type variable 을 받게 하면 새로운 문제가 생기는데 type class 정의에서 우리는 알고 있지만 컴파일러는 모르는 정보가 생긴다는 점입니다. 그건 바로 Collection 의 type 이 해당 Collection 의 원소의 type 을 결정한다는 정보입니다. 무슨 말이냐하면 어떤 Collection 의 type 이 [a] 꼴이면 그것의 원소의 type 은 a 가 된다는 것입니다. 예를 하나 더 들어보면 Collection 의 type 이 Hashmap a 이면 그것의 원소의 type 은 a 가 되는 것이 자명합니다. 우리는 이렇듯 type 사이에 관계가 있다는 정보를 알고 있지만, 우리가 Collection type class 를 정의한 코드에는 이것에 대한 정보가 없기 때문에 compiler 는 이에 대한 정보를 알지 못한 상황이 됩니다. 그 결과 Compiler 는 필요 이상으로 일반화된 type 의 함수를 추론하게 됩니다. 예를 들어 Collection 에 원소를 두 개 추가하는 다음과 같은 함수를 정의했다고 합시다.
```haskell
ins2 xs a b = insert (insert xs a) b
```
이 함수의 type 을 컴파일러가 어떻게 추론했는지 확인해보면 다음과 같은 꼴로 추론함을 확인할 수 있습니다.

    > :t ins2
    ins2::(Collection c e1, Collection c e) => c -> e1 -> e -> c

이는 우리가 원하는 결과가 아닙니다. e1 과 e 가 같은 type 이라는 것을 compiler 가 모르기 때문에 이처럼 지나치게 일반화된 type 으로 추론을 했습니다. 이 같은 문제를 해결할 수 있는 것이 다음의 Functional Dependency 확장입니다.
####FunctionalDependencies
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


####RecordWildCards
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
####ParallelListComp
List comprehension 에서는 Cartesian product 가 나옵니다. 즉, [x+y|x<-[1..3],y<-[10..12]] 의 결과는 길이가 9인 List 가 됩니다. ParallelListComp 확장을 쓰면 각 원소들을 1:1 대응하여 연산을 수행합니다. ParallelListComp 확장의 경우 generator 간 구분은 쉼표가 아니라 수직선으로 합니다.
```haskell
{-# LANGUAGE ParallelListComp #-}
[x+y|x<-[1..3] | y <-[10..12]] -- 결과는 [11,13,15]
```
이는 zipWith (+) [1..3] [10..12] 한 것과 같은 결과로서 ParallelListComp 를 이용한 표현식은 zipWith 를 이용하여 똑같이 작성할 수 있습니다. 그럼에도 ParallelListComp 확장을 쓰면 좋은 점은 코드를 좀 더 보기좋게 작성할 수 있다는 점에 있습니다.
####TransformListComp
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
####FlexibleContexts
이 확장의 대표적인 사용예는 다음의 class constraints 작성 규칙 완화입니다.

    (Stream s u Char) =>

즉, type variable 을 polymorphic 하게 사용하지 않고 특정 type 으로 지정할 수 있습니다. 여기서는 Char.

FlexibleContexts 확장은 Instance contexts, Class contexts 그리고 Typeclass constraints 를 작성할 때 더 유연하게 코드를 작성할 수 있게 합니다. 각각의 경우를 살펴보겠습니다.

첫째, Instance contexts 입니다. Haskell 표준에서 Instance contexts 는 "C a" 꼴만 가능합니다. 여기서 C 는 임임의 typeclass 이름이고 a 는 type variable 입니다. 그런데 FlexibleContexts 확장을 쓰면 다음과 같이 훨씬 더 자유롭게 Instance 선언을 할 수 있습니다.
```haskell
instance C Int [a]          -- Multiple parameters
instance Eq (S [a])         -- Structured type in head

-- Repeated type variable in head
instance C4 a a => C4 [a] [a]
instance Stateful (ST s) (MutVar s)

-- Head can consist of type variables only
instance C a
instance (Eq a, Show b) => C2 a b

-- Non-type variables in context
instance Show (s a) => Show (Sized s a)
instance C2 Int a => C3 Bool [a]
instance C2 Int a => C3 [a] b
```

둘째, Class contexts 입니다. 앞서 Instance contexts 작성에 대한 규칙을 완화한 것처럼 이번에는 Class contexts 작성에 관한 규칙을 완화하고 있습니다.
```haskell
class Functor (m k) => FiniteMap m k where
  ...

class (Monad m, Monad (t m)) => Transform t m where
  lift :: m a -> (t m) a
```

셋째, type signature 작성시 typeclass constraints 작성에 관한 규칙을 완화합니다. 아래 코드를 보면 Eq [a] 처럼 type variable 자리에 list of type variable 이 들어가 있습니다. 또한 Ord (T a ()) 처럼 type constructor 가 들어있습니다.
```haskell
g :: Eq [a] => ...
g :: Ord (T a ()) => ...
```
####RecursiveDo
Haskell 에서는 lazy evaluation 덕분에 다음과 같은 순환 구조의 재귀코드를 작성할 수 있습니다.
```haskell
main = print $
  let x = fst y
      y = (3, x)
  in snd y
```
그런데 do 블럭 안에서는 그렇게 할 수 없습니다. 예를 들어 다음 코드는 에러가 납니다.
```haskell
import Control.Monad.Identity

main = print((
  do x <- return $ fst y
     y <- return (3, x)
     return $ snd y)::Identity Integer)
```

    a.hs:4:24: error: Variable not in scope: y :: (Integer, b0)

RecursiveDo 확장을 쓰면 do 블럭 안에서도 순환구조의 재귀 코드를 쓸 수 있습니다. 다음 코드에서 rec 은 RecursiveDo 확장 사용에 따른 예약어입니다.
```haskell
{-# LANGUAGE RecursiveDo #-}
import Control.Monad.Identity

main = print((
  do rec x <- return $ fst y
         y <- return (3, x)
     return $ snd y)::Identity Integer)
```

####NoMonomorphismRestriction
먼저 MonomorphismRestriction 이 무엇인지 알아봅시다. 일단 Monomorphism 이란 Polymorphism 과 반대 개념입니다. 다음 코드를 파일로 저장한 다음 GHCi 에서 load 해 봅시다.
```haskell
-- Mono.hs
plus = (+)
```

    > :l Mono.hs
    [1 of 1] Compiling Main    ( Mono.hs, interpreted )
    Ok, modules loaded: Main.
    > :t plus
    plus :: Integer -> Integer -> Integer
    > plus 1.1 2

    <interactive>:3:6: error:
        • No instance for (Fractional Integer)
            arising from the literal ‘1.1’
        ...

위에서 보듯 plus 함수의 type 은 (+) 연산자의 type 인 Num a => a -> a -> a 와는 달리 polymorphic 하지 않습니다. 그렇기에 plus 1.1 2 같은 코드는 error 가 납니다. 이렇게 되는 이유는 ghc 컴파일러는 MonomorphismRestriction 이 기본 설정이기 때문입니다. 반면 GHCi 에서는 NoMonomorphismRestriction 이 기본 설정이어서 똑같이 plus = (+) 를 정의해도 이것의 type 이 (+) 의 type 과 같습니다.

    > let plus = (+)
    > :t plus
    plus :: Num a => a -> a -> a

NoMonomorphismRestriction 이 뜻하는 바는 가능한 한 최대로 polymorphic type 으로 추론하라는 것이고 MonomorphismRestriction 은 그 반대의 뜻입니다. 따라서 ghc 컴파일에서도 최대한 polymorphic type 으로 type inference 가 되도록 하려면 NoMonomorphismRestriction 을 사용하면 됩니다. 다음처럼
```haskell
{-# LANGUAGE NoMonomorphismRestriction #-}
plus = (+)
```

####DeriveFunctor, DeriveFoldable, DeriveTraversable
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

####DeriveGeneric, DeriveAnyClass
위에서 사용한 DeriveFunctor, DeriveFoldable, DeriveTraversable 은 모두 base 라이브러리에 속한 Functor, Foldable, Traversable 의 Instance 를 손쉽게 만들수 있게 해주었습니다. 그렇다면 base 라이브러리에 속하지 않은 type class 의 Instance 를 이와 같은 방식으로 손쉽게 만들 수 있는 확장은 없을까요? DeriveGeneric 확장이 바로 이 같은 상황에서 쓸 수 있는 확장입니다. Data.Aeson 모듈을 이용하여 원하는 데이터를 JSON 형식으로 바꾸는 코드를 작성해보겠습니다.
```haskell
-- Jedi.hs
{-# LANGUAGE DeriveGeneric #-}
import Data.Aeson
import Data.Aeson.Types
import GHC.Generics

data Jedi = Jedi{name::String, age::Int, greeting::String} deriving (Show, Generic)
instance FromJSON Jedi
instance ToJSON Jedi

jediAsJSON = encode (Jedi{age=900, name="Yoda", greeting="May the Lambda be with you."})
decodedJedi:: Maybe Jedi
decodedJedi = decode jediAsJSON::Maybe Jedi
```

    > :l Jedi.hs
    Ok, modules loaded: Main
    > print jediAsJSON
    "{\"age\":900,\"name\":\"Yoda\"}"
    > print decodedJedi
    Just (Jedi {name = "Yoda", age = 900})

한편, DeriveAnyClass 확장을 함께 쓰면 더 코드를 간결하게 작성할 수 있습니다. 위 코드에서 instance 선언부가 필요없습니다.

```haskell
{-# LANGUAGE DeriveGeneric, DeriveAnyClass #-}
...
data Jedi = Jedi{name::String, age::Int, greeting::String} deriving (Show, Generic, FromJSON, ToJSON)

jediAsJSON = encode (Jedi{age=900, name="Yoda", greeting="May the Lambda be with you."})
...
```

####DeriveDataTypeable
이 확장은 Dynamic typing 에 관한 것입니다. Haskell 은 Static typing 을 취하고 있지만 때때로 Dynamic typing 이 필요한 경우가 있을 때 이 확장을 사용합니다.
#####Data.Typeable

####GeneralizedNewtypeDeriving
newtype 을 써서 만든 자료형은 deriving 방식을 사용하여 특정 type classe 의 instance 로 만들 수 없는데, GeneralizedNewtypeDeriving 확장은 그걸 할 수 있게 해줍니다.
```haskell
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
newtype Dollars = Dollars Int deriving (Eq, Show, Num)
a = (Dollars 8) + (Dollars 9) -- Dollars 17
```

## 첫날 두번째 100분
다음의 ghc 컴파일러 확장을 배웁시다.
- [RankNTypes](#rankntypes)
- [GADTs(Generalised Algebraic Data Types)](#gadtsgeneralised-algebraic-data-types)
- [KindSignatures, DataKinds](#kindsignatures-datakinds)
- [PolyKinds](#polykinds)
- [ScopedTypeVariables](#scopedtypevariables)
- [ExistentialQuantification](#existentialquantification)
    * [Existential Types](#existential-types)
- [TypeFamilies](#typefamilies)
- [TypeFamilyDependencies](#typefamilydependencies)
    * [Injective Type Families](#injective-type-families)
- [TypeInType](#typeintype)
- [TypeOperators](#typeoperators)
- [LiberalTypeSynonyms](#liberaltypesynonyms)
- [DefaultSignatures](#defaultsignatures)
- [ImplicitParams](#implicitparams)
- [ConstraintKinds](#constraintkinds)

####RankNTypes
Haskell 에서의 type 은 기본적으로 Rank-1 type 입니다. 그렇다면 Rank-2 type 이란 것도 있는가? 있습니다. 이 Rank-N type 에 대해 알려면 우선 forall 예약어에 대해 알아야 합니다. 많이 쓰는 함수 length 의 typ 은 다음과 같습니다.
```haskell
length:: [a] -> Int
```
그런데 length 함수의 type 에서 사실 생략된 부분이 있습니다. 그 생략된 부분을 드러내보면 length 함수의 type 은 다음과 같습니다.
```haskell
length:: forall a.[a] -> Int
```
Haskell 에서 polymorphic 함수의 type 에는 이처럼 forall 이 묵시적으로 붙어 있습니다. forall 이 뜻하는 바는 a 의 type 이 무엇이건간에 [a] -> Int 꼴 함수 length 가 성립한다는 것입니다. 즉, a 가 Int 여도 되고 Char 이어도 되고 Bool 이어도 되고 등등. 참고로 이것은 Predicate Logic 의 Universal Quantification 와 같은 것입니다. 예) 모든 사람은 죽는다: ∀x{Human(x) -> Die(x)}
Polymorphic type system 에서 Rank 라 함은 이 forall 이 type 표기의 어느 부분에 나올지에 관한 것입니다. Rank-1 type system 이라면 forall 은 type 표기의 가장 바깥쪽에만 올 수 있습니다. 위의 length 함수의 type 처럼. Rank-2 type system 이라면 forall 이 type 표기의 한 단계 안쪽까지 올 수 있습니다. 다음은 Rank-2 type 과 Rank-3 type 의 예입니다.
```haskell
{-# LANGUAGE RankNTypes #-}
someInt:: (forall a.a->a) -> Int -- Rank-2 type
someInt id' = id' 3

someOtherInt::((forall a.a->a)->Int) -> Int -- Rank-3 type
someOtherInt someInt' = someInt' id + someInt' id
```
Rank-1 type 인 경우 당연히 가장 바깥쪽에 forall 이 존재하는 것이 자명하기 때문에 forall 을 쓰는 것이 생략되어 있습니다. 하지만 Rank-2 type 같은 경우 전체 type 의 일부분에 Rank-1 type 함수가 포함된 형태(즉, 다른 함수를 인자로 받는 함수인데 인자로 받는 함수가 Rank-1 type 함수)이기 때문에 이것을 나타내기 위해서는 forall 예약어를 명시적으로 써야만 합니다.

이제 RankNTypes 함수를 만들어보겠습니다. 다음과 같이 동작하는 함수를 만들려고 합니다.
```haskell
applyToTuple f (x,y) = (f x, f y)
-- applyToTuple length ("hello",[1,2,3]) 의 결과는 (5,3)
```
코드를 이렇게 작성하고 applyToTuple length ("hello",[1,2,3]) 을 실행하면 다음과 같은 에러가 납니다.

    <interactive>:2:31: error:
    • No instance for (Num Char) arising from the literal ‘1’
    • In the expression: 1
      In the expression: [1, 2, 3]
      In the second argument of ‘applyToTuple’, namely
        ‘("hello", [1, 2, 3])’

위 에러가 뜻하는 바는 Num 이 Char 의 Instance 가 아니라는 것입니다. 왜 이런 에러가 나는지 알아보기 위해 Haskell 컴파일러가 applyToTuple 함수의 type 을 어떻게 추론하는지 확인해봅시다.

    > :t applyToTuple
    applyToTuple :: (t1 -> t) -> (t1, t1) -> (t, t)

applyToTuple 함수는 두번째 인자로 같은 type 을 요소로 갖는 tuple 을 받게 되어 있습니다. 따라서 위 예시에서는 tuple 의 첫번째 요소의 type 이 Char 였기에 두번째 요소 역시 type 이 Char 로 설정되었는데 막상 Num 이 들어왔기에 에러가 난 것입니다. 즉, applyToTuple 의 함수의 type 이 우리가 기대하는 바와는 다르게 추론이 되었습니다. 그렇다면 applyToTuple 함수의 type 을 명시적으로 applyToTuple::([a]->Int)->([b],[c])->(Int,Int) 로 주면 어떻게 될까요? 쉽게 예상할 수 있듯이 컴파일할 때 type error 가 납니다. type 'b' 와 type 'a' 가 서로 맞지 않고, type 'c' 와 type 'a' 역시 서로 맞지 않다는 error 가 납니다. 직접 해보시기 바랍니다.

applyToTuple 함수에 우리가 원하는 대로 type 을 주기 위해서는 RankNTypes 확장을 써야 합니다. 다음처럼 말입니다.
```haskell
{-# LANGUAGE RankNTypes #-}
applyToTuple:: (forall a.[a]->Int) -> ([b],[c]) -> (Int, Int)
applyToTuple f (x,y) = (f x, f y)
```
이제 이 함수를 실행하면 바라던 방식대로 동작하는 것을 볼 수 있습니다.

    > applyToTuple length ("hello", [1,2,3])
    (5,3)

RankNTypes 확장이 필요한 이유는 곰곰히 생각해 보면 이는 Polymorphic function, 특히 Haskell 에서 구현하고 있는 Parametric polymorphism 에 대한 이해와 연결됩니다. 앞서 예를 들었던 length 함수의 type "[a] -> Int" 가 뜻하는 바는 [Char] -> Int 꼴이나 [Double] -> Int, [String] -> Int 등의 여러 함수의 집합으로 볼 수 있다는 것입니다. 즉, 실제로 length 함수가 사용될 때 length 함수의 type 이 결정된다는 것입니다. 다음 코드를 봅시다.

```haskell
a = [1,2,3]::[Integer]
length a -- 여기서의 length 함수의 type 은 [Integer] -> Int 입니다.
b = [1,2,3]::[Double]
length b -- 여기서의 length 함수의 type 은 [Double] -> Int 입니다.
```
즉, Polymorphic function 의 type 은 해당 함수가 쓰이는 시점에 정해지는(instantiated) 것이지요. 이를 applyToTuple 예제에 대해서 생각해 보면 applyToTuple 함수의 첫번째 인자인 함수 f 의 type 은 "hello" 에 대해 적용될 때는 [Char]->Int 로 정해지고 [1,2,3] 에 대해 적용될 때는 Num a => [a] -> Int 로 정해집니다. 그렇기 때문에 forall 예약어의 위치가 중요한 것입니다.

이처럼 parametric polymorphism 에서는 type variable 이 함수의 동작을 크게 규정합니다. 이를 Parametricity 라고 부르는데 예를 들어 f::[a] -> [a] 꼴인 함수 f 가 있을 때 이 함수가 하는 일을 추측해봅시다. 언뜻 매우 다양한 함수가 이 함수꼴 집합에 포함될 것이라고 생각할 수 있으나 사실은 정반대입니다. 모든 type 에 대하여 고려를 해야 하기 때문에 [a] -> [a] 꼴 함수집합에 속할 수 있는 함수는 매우 제한적입니다. 예를 들어 이 함수가 각 인자를 1 만큼 증가시키는 함수라고 추측해봅시다. 가능할까요? type variable 'a' 가 Int type 이면 가능합니다. 그런데 Bool type 이라면? 불가능한 일입니다. 따라서 [a] -> [a] 꼴 함수가 할 수 있는 일은 인자들의 순서를 재배열하거나, 인자들의 갯수를 늘리거나 또는 줄이는 일 정도입니다. 그 외에 혹시라도 뭐가 또 있을 수 있을까요?

####GADTs(Generalized Algebraic Data Types)
다음과 같은 data type 을 정의한다고 해 봅시다.
```haskell
data Expr = I Int
          | B Bool
          | Add Expr Expr
          | Mul Expr Expr
          | Eq Expr Expr
```
이 data type 은 다음과 같은 expression 으로 표현이 가능합니다.

    (I 5 `Add` I 1) `Eq` I 7 :: Expr

그런데 다음도 역시 유효한 expression 입니다.

    B True `Add` I 5 :: Expr

이러한 상황을 방지하고 싶으면 어찌하면 될까요? 다시 말해 type safety 를 확보하고 싶다면? 그래서 등장하는 것이 GADT 확장입니다. 우선 GADT 의 문법을 살펴보겠습니다. Generalised Abstract Data Type 의 선언은 보통의 data type 선언과는 다른 형식의 문법을 사용합니다. 다음 코드를 봅시다.
```haskell
{-# LANGUAGE GADTs #-}
-- data Maybe' a = Nothing' | Just' a
data Maybe' a where
  Nothing':: Maybe' a
  Just':: a -> Maybe' a

-- data List a = Nil | Cons a (List a)
data List a where
  Nil:: List a
  Cons:: a -> List a -> List a

-- data Bool' = True' | False'
data Bool' where
  True':: Bool'
  False':: Bool'
```
data type 선언을 마치 type class 선언을 하는 것처럼 합니다. 물론 위의 data type 들은 GADT 는 아닙니다. 단지 GADT 선언 문법을 사용하고 있을 뿐입니다. 위의 data type 들은 선언문법이 다를뿐 기존 방식으로 선언했을 때의 data type 과 완전히 같습니다.

이제 실제로 GADT 선언 문법을 사용했을 때 추가로 더 할 수 있는 일이 무엇인지 알아봅시다. 다음 코드를 봅시다.
```haskell
{-# LANGUAGE GADTs #-}
data G a where
  MkGInt:: G Int
  MkGBool:: G Bool
```
G a 라는 자료형의 Constructor 두 개가 서로 다른 type 입니다! MkGInt 의 type 은 G Int 이고 MkgBool 의 type 은 G Bool 로 서로 다릅니다! 이게 어떤 장점이 있는지 알아봅시다. 다음 코드를 봅시다.
```haskell
{-# LANGUAGE GADTs #-}
data Natural = Zero | Succ Natural
data Vec n a where
  VNil:: Vec Zero a
  VCons:: a -> Vec n a -> Vec (Succ n) a
infixr 5 `VCons` -- VCons 연산자를 right-associative 하게 정의.
```
위의 Vec n a 자료형에서 첫번째 인자 n 은 Vector 의 길이를 뜻하고 두번째 인자 a 는 Vector 요소의 type 을 뜻합니다.

이때 다음처럼 dummy type variable(또는 phantom)을 이용합니다. 아래 코드에서 a 가 phantom 입니다. a 는 where 구문 이후에 전혀 등장하지 않기 때문에 phantom 이라고 부릅니다.
```haskell
data Expr a where
      I:: Int -> Expr Int
      B:: Bool -> Expr Bool
      Add:: Expr Int -> Expr Int -> Expr Int
      Mul:: Expr Int -> Expr Int -> Expr Int
      Eq:: Expr Int -> Expr Int -> Expr Bool
```
이렇게 하면 type safety 를 확보할 수 있습니다. 그리고 다음과 같은 smart constructor 를 만들 수 있습니다.
```haskell
add:: Expr Int -> Expr Int -> Expr Int
add = Add
eq:: Expr Int -> Expr Int -> Expr Bool
eq = Eq
```
GADTs extension 을 통해 이러한 것이 가능합니다.

####KindSignatures, DataKinds
Haskell 은 type variable 의 kind 를 알아서 유추하지만 프로그래머가 직접 kind 를 명시해주는 것이 코드를 이해하기에 좋을 수도 있습니다. 마치 함수의 type 을 프로그래머가 직접 명시해 주는 것처럼. 다음 코드를 봅시다.
```haskell
{-# LANGUAGE GADTs #-}
data Z
data S n

data Vec n a where
  VNil:: Vec Z a
  VCons:: a -> Vec n a -> Vec (S n) a
```
위 코드에서 Vec 의 kind 는 * -> * -> * 입니다. 이를 명시적으로 주려면 다음처럼 하면 됩니다.
```haskell
{-# LANGUAGE GADTs, KindSignatures #-}
data Z
data S n

data Vec:: * -> * -> * where
  VNil:: Vec Z a
  VCons:: a -> Vec n a -> Vec (S n) a
```
그런데 위 코드의 Vec 자료형의 의도는 type 자체에 해당 자료형의 크기를 포함하는 것입니다. 즉, 길이가 1인 Vec 자료형은 type 이 Vec (S Z) a 이고 길이가 2면 type 이 Vec (S (S Z)) a 가 되어서 type 자체에 자료형의 길이가 드러나게 되는 것입니다. 따라서 Vec type 의 kind 는 우리의 의도를 더 잘 표현하려면 * -> * -> * 보다는 좀 더 구체적으로 (자연수 type의 kind) -> * -> * 가 되는 것입니다. 일단 자연수를 뜻하는 type 을 다음처럼 정의하겠습니다. 참고로 이는 앞서 GADTs 에 대해 배울 때 한 번 나왔습니다.
```haskell
data Natural = Zero | Succ Natural
```
그런데 이 Natural type 의 kind 가 * 가 아니라 다른 것과 구분할 수 있는 것이어야 합니다. 이 때 필요한 것이 DataKinds 확장입니다. DataKinds 확장을 쓰면 자료형 선언시 해당 자료형 type 의 kind 가 자동으로 만들어집니다. 위 코드를 ghci 에서 불러온 다음 DataKinds 확장을 활성화해보겠습니다.

    > :set -XDataKinds
    > :k 'Zero
    'Zero:: Natural
    > :k 'Succ
    'Succ:: Natural -> Natural

DataKinds 확장을 하면 data type 은 kind 로, value constructor 는 type constructor 로 승격이 됩니다. 다시 말해 Natural 이란 data type 이 이제는 kind 를 뜻하기도 하는 것이지요. 그리고 이렇게 승격을 통해 만들어진 type constructor 들은 원래의 value constructor 이름 앞에 홑따옴표가 붙게 됩니다. 이제 Natural 은 kind 이기도 하므로 다음과 같은 코드를 작성할 수 있습니다.

```haskell
{-# LANGUAGE GADTs, KindSignatures, DataKinds #-}
data Natural = Zero | Succ Natural

data Vec:: Natural -> * -> * where
  VNil:: Vec Zero a
  VCons:: a -> Vec n a -> Vec (Succ n) a
```
####PolyKinds
다음과 같은 코드가 있습니다.
```haskell
data App f a = MkApp (f a)
```
App type 의 kind 를 확인해봅시다. ghci 에서 해 보겠습니다.

    > data App f a = MkApp (f a)
    > :k App
    App :: (* -> *) -> * -> *

함수로 보이는 f 의 kind 가 * -> * 인 것이지요. 그런데 위의 코드를 보면 알 수 있듯이 함수 f 의 인자는 App type 선언시 두번째 인자 a 와 같습니다. 따라서 (\* -> \*) -> * -> * 보다는 (k -> \*) -> k -> * 가 더 정확한 kind 입니다. 이렇게 좀 더 구체적으로 kind 를 유추할 수 있게 하려면 PolyKinds 확장이 필요합니다. 이 확장은 Kind Polymorphism 을 지원한다는 뜻입니다. Type Polymorphism 을 Kind 에도 적용하는 것입니다. 이제 ghci 에서 :set -XDataKinds 를 설정하고 다시 해 보겠습니다.

    > :set -XPolyKinds
    > data App f a = MkApp (f a)
    > :k App
    App :: (k -> *) -> k -> *

이처럼 좀 더 구체적으로 kind 를 유추함을 알 수 있으며 이로 인해 App 을 좀 더 다양한 kind 에서 쓸 수 있습니다.

####ScopedTypeVariables
이 확장은 "Lexically" scoped type variable 에 관한 것입니다. 다음 코드를 봅니다.
```haskell
f :: [a] -> [a]
f (x:xs) = xs ++ [ x :: a ]
```
위 코드는 함수 f 의 선언부분에서 x::a 라는 type signature 를 주고 있습니다. 이렇게 하는 의도는 함수 f 의 선언부에 등장하는 x 라는 변수가 함수 f 의 type signature 에서 나오는 a 와 같은 type 의 원소 임을 명시해주기 위함입니다. 코드 이해를 돕기 위한 문서화라고 할 수 있습니다. 그런데 이 코드는 에러가 납니다. 꽤 긴 에러가 나는데 그 중에서 눈여겨 볼 부분은 다음입니다.

    • Couldn't match expected type ‘a1’ with actual type ‘a’
      ‘a’ is a rigid type variable bound by
        the type signature for:
          f :: forall a. [a] -> [a]
        at lexically.hs:1:6
      ‘a1’ is a rigid type variable bound by
        an expression type signature:
          forall a1. a1
        at lexically.hs:2:25

이를 보면 a1 이 변수 x 의 type 이라고 나옵니다. 우리가 작성한 코드에서는 분명 변수 x 의 type 을 a 라고 주었는데 갑자기 a1 이라고 나옵니다. 즉, 이 에러가 말하는 바는 함수 f 의 signature 에서는 type 이 a 라고 했는데 함수 정의부에는 a1 이라는 다른 type 이니 서로 type 이 맞지 않다는 것입니다. 컴파일러가 이렇게 추론하는 이유는 Haskell2010 표준은 type signature 의 type variable 의 범위를 type signature 로 한정하고 있기 때문입니다. 즉, type signature 의 a 라는 type variable 을 함수 f 의
정의부에서는 전혀 알 수가 없습니다. ScopedTypeVariable 확장은 말 그대로 type variable 이 영향을 미치는 범위를 type signature 이상으로 확장해줍니다. 위 코드를 이 확장을 써서 다시 작성하면 다음과 같습니다.
```haskell
{-# LANGUAGE ScopedTypeVariables #-}
f :: forall a. [a] -> [a]
f (x:xs) = xs ++ [ x :: a ]
```

참고로 이 확장에 대한 원 논문은 Simon Peyton Jones 가 작성한 [Lexically-scoped type variables](https://www.microsoft.com/en-us/research/publication/lexically-scoped-type-variables/) 입니다.

####ExistentialQuantification
앞서 Predicate Logic 의 Universal Quantification 에 대해 잠시 다루었는데 Existential Quantification 은 다음과 같습니다. 예) 똑똑한 한국사람이 적어도 한명 있다: ∃x{Korean(x) ∧ Smart(x)}

Universal Quantification 과 Existential Quantification 은 서로 상호 변환이 가능한데 이 때 다음 두 가지 추론 규칙을 사용합니다.

    α ⇒ β ≡ ¬α ∨ β    (Implication Elimination)
    ∀x.A(x) ≡ ¬∃x.¬A(x)    (De Morgan's rules)

위 두 규칙을 이용하여 (∀x.A(x) ⇒ B) 을 바꾸어보겠습니다.

    (∀x.A(x) ⇒ B)
    (∀x.¬A(x) ∨ B)
    (∀x.¬A(x)) ∨ B
    (¬∃x.A(x)) ∨ B
    (∃x.A(x)) ⇒ B

이제 Haskell 코드에서 이것이 어떻게 나타나는지 살펴보겠습니다. Haskell 의 List 는 Homogeneous list 인데 여기서 한 번 Heterogenous list 를 만들어보겠습니다. 다음처럼 ShowBox 라는 wrapper 를 만들어서 하면 됩니다.
```haskell
{-# LANGUAGE ExistentialQuantification #-}
data ShowBox = forall s. Show s => SB s

heteroList :: [ShowBox]
heteroList = [SB (), SB 5, SB True]
```
Existential Quantification 을 이용하여 세 개의 서로 다른 type 을 하나의 list 에 담았습니다. 위 코드에서 forall 예약어를 쓴 부분이 바로 Existential Quantification 확장을 이용하는 부분입니다. 이 부분은 위의 Predicate logic 설명에 나온 (∀x.A(x) ⇒ B) 을 코드로 옮긴 것입니다.

이렇게 만든 heteroList 를 사용하는 코드를 만들어보겠습니다. ShowBox 가 감싸고 있는 것은 Show typeclass 에 속하기 때문에 show 함수를 쓸 수 있습니다.
```haskell
instance Show ShowBox where
  show (SB s) = show s

f :: [ShowBox] -> IO ()
f xs = mapM_ print xs

main = f heteroList
```
위 코드를 컴파일하고 실행해보면 heteroList 에 담겨있는 것을 출력함을 볼 수 있습니다.

Existential Quantification 은 그자체로는 특별한 쓰임새가 있지는 않으나 다른 기능들의 밑바탕에 깔리는 중요 개념입니다. 따라서 Existential type 에 대하여 이해를 할 필요가 있습니다.

#####Existential Types
Existential type 은 Abstract Data Type(이후 ADT) 을 위한 것입니다.

####TypeFamilies
Type families 확장은 다음 네 가지 개념을 포함합니다.

첫째, **Associated (Data) Type**
```haskell
class ArrayElem e where
  data Array e
  index :: Array e -> Int -> e

instance ArrayElem Int where
  data Array Int = IntArray UIntArr
  index (IntArray a) i = ...
```
이 코드를 보면 ArrayElem typeclass 안에 Array e 라는 data type 이 선언되어 있습니다. 이것이 표현하려는 바는 Array e 라는 data type 이 ArrayElem typeclass 에 종속적인 관계라는 것입니다. 

둘째, **Associated (Type) Synonym**
```haskell
class Collection c where
  type Elem c
  insert :: Elem c -> c -> c

instance Eq e => Collection [e] where
  type Elem [e] = e
  ...
```
이번에는 data type 대신 type synonym 이 나올 뿐 첫번째 경우와 같은 맥락입니다.

위의 두 가지 개념은 "Associated" 란 개념으로 묶을 수 있는데 이는 표준 typeclass 정의 안에 data type 또는 type synonym 을 두는 것입니다. 그리고 아래에 나오는 두 가지 개념은 "Family" 란 개념으로 묶을 수 있는데 이는 "Associated" 란 개념을 엄밀하게 일반화한 것입니다. 반대로 말해 "Associated" 개념은 "Family" 란 개념의 syntactic sugar 라고 할 수 있습니다.

셋째, **Data (Type) Family**
```haskell
data family Array e
data instance Array Int = IntArray UIntArr
data instance Array Char = MyCharArray a b
```
family 라는 예약어를 사용합니다.

넷째, **(Type) Synonym Family**
```haskell
type family Elem c
type instance Elem [e] = e
type instance Elem BitSet = Char
```

먼저 "Associated" 개념을 사용하는 예를 보겠습니다. 다음 코드는 포켓몬을 표현하고 있습니다. (https://www.schoolofhaskell.com/school/to-infinity-and-beyond/pick-of-the-week/type-families-and-pokemon 의 설명을 가져왔습니다)
```haskell
{-# LANGUAGE MultiParamTypeClasses #-}
-- Names of Pokemon
data Fire = Charmander | Charmeleon | Charizard deriving Show
data Water = Squirtle | Wartortle | Blastoise deriving Show
data Grass = Bulbasaur | Ivysaur | Venusaur deriving Show
-- Moves of Pokemon
data FireMove = Ember | FlameThrower | FireBlast deriving Show
data WaterMove = Bubble | WaterGun deriving Show
data GrassMove = VineWhip deriving Show

class (Show pokemon, Show move) => Pokemon pokemon move where
  pickMove :: pokemon -> move

instance Pokemon Fire FireMove where
  pickMove Charmander = Ember
  pickMove Charmeleon = FlameThrower
  pickMove Charizard = FireBlast

instance Pokemon Water WaterMove where
  pickMove Squirtle = Bubble
  pickMove _ = WaterGun

instance Pokemon Grass GrassMove where
  pickMove _ = VineWhip
```
위 코드를 ghci 에서 불러들인 다음 print (pickMove Charmander) 를 실행해보세요. Type error 가 납니다. 그 이유는 pickMove Charmander 코드를 보고 type checker 가 *Pokemon Fire a* 라는 instance 를 찾는데 해당 instance 가 없기 때문입니다. 그래서 이러한 type error 를 피하려면 다음처럼 type annotation 을 주어야 합니다.

    > print (pickMove Charmander::FireMove)
    Ember

이렇게 하면 type checker 는 *Pokemon Fire FireMove* instance 를 찾게 되고 이것이 있기 때문에 type error 가 나지 않습니다. 즉, Fire 형 포켓몬은 FireMove type 의 move 를 사용한다는 것을 type 차원에서 표현하지 못하고 있기 때문에 이렇게 코드 작성시 type annotation 으로 알려주어야 하는 것이지요.

이번에는 포켓몬간 결투를 코드에 추가해보겠습니다.
```haskell
import Data.Tuple (swap)
{- 앞서 나온 코드를 이 부분에 붙입니다
-}
printBattle :: String -> String -> String -> String -> String -> IO ()
printBattle pokemonOne moveOne pokemonTwo moveTwo winner = do
  putStrLn $ pokemonOne ++ " used " ++ moveOne
  putStrLn $ pokemonTwo ++ " used " ++ moveTwo
  putStrLn $ "Winner is: " ++ winner ++ "\n"

class (Pokemon pokemon move, Pokemon foe foeMove) => Battle pokemon move foe foeMove where
  battle :: pokemon -> foe -> IO ()
  battle pokemon foe = do
    printBattle (show pokemon) (show move) (show foe) (show foeMove) (show pokemon)
    where
      move = pickMove pokemon
      foeMove = pickMove foe

instance Battle Water WaterMove Fire FireMove
instance Battle Fire FireMove Water WaterMove where
  battle a b = fmap swap $ battle b a

instance Battle Grass GrassMove Water WaterMove
instance Battle Water WaterMove Grass GrassMove where
  battle a b = fmap swap $ battle b a

instance Battle Fire FireMove Grass GrassMove
instance Battle Grass GrassMove Fire FireMove where
  battle a b = fmap swap $ battle b a
```
이 코드를 다시 ghci 에서 불러들인 다음 battle Squirtle Charmander 코드를 실행해봅니다. 이 역시 에러가 납니다. 앞서와 마찬가지로 battle Squirtle Charmander::IO (WaterMove, FireMove) 로 type annotation 을 주어서 호출하면 에러없이 동작합니다.

    > battle Squirtle Charmander::IO (WaterMove, FireMove)
    Squirtle used Bubble
    Charmander used Ember
    Winner is: Squirtle

    (Bubble,Ember)

이렇게 type annotation 을 주어야만 프로그램이 동작하는 이유는 type checker 가 Pokemon type 과 Pokemon move type 간의 관계를 알지 못하기 때문입니다. 따라서 Pokemon Fire WaterMove 와 같은 전혀 바라지 않은 instance 도 만들 수 있습니다. TypeFamilies 확장을 이용하여 이러한 것을 보완할 수 있습니다. 다음 코드를 보면 Pokemon typeclass 가 인자를 하나만 받게 되어 있고 대신 Move a 라는 associated type 이 들어가 있습니다. 즉, 이제부터는 FireMove 대신 Move Fire 를 사용하는 것으로 Pokemon type 과 Pokemon move type 간의 연관성이 type 수준에서 생기는 것입니다.
```haskell
{-# LANGUAGE TypeFamilies, FlexibleContexts #-}
class (Show a, Show (Move a)) => Pokemon a where
  data Move a :: *
  pickMove :: a -> Move a
```
위의 type class 정의를 이용해서 코드를 다시 작성해 보면 다음과 같습니다.
```haskell
{-# LANGUAGE TypeFamilies, FlexibleContexts #-}
class (Show a, Show (Move a)) => Pokemon a where
  data Move a :: *
  pickMove :: a -> Move a

data Fire = Charmander | Charmeleon | Charizard deriving Show
instance Pokemon Fire where
  data Move Fire = Ember | FlameThrower | FireBlast deriving Show
  pickMove Charmander = Ember
  pickMove Charmeleon = FlameThrower
  pickMove Charizard = FireBlast

data Water = Squirtle | Wartortle | Blastoise deriving Show
instance Pokemon Water where
  data Move Water = Bubble | WaterGun deriving Show
  pickMove Squirtle = Bubble
  pickMove _ = WaterGun

data Grass = Bulbasaur | Ivysaur | Venusaur deriving Show
instance Pokemon Grass where
  data Move Grass = VineWhip deriving Show
  pickMove _ = VineWhip
```
위 코드를 ghci 에서 불러들인 다음 pickMove Charmander 를 실행하면 이번에는 type annotation 없이도 코드가 정상적으로 실행함을 알 수 있습니다. 이제 Battle type class 도 새로운 Pokemon type class 에 맞추어 수정을 해보겠습니다.
```haskell
{-# LANGUAGE TypeFamilies, MultiParamTypeClasses, FlexibleContexts #-}
{- 바로 전의 Pokemon type class 관련 코드를 여기 붙입니다.
-}
class (Pokemon a, Pokemon b) => Battle a b where
  battle :: a -> b -> IO ()
  battle pokemon foe = do
    printBattle (show pokemon) (show move) (show foe) (show foeMove) (show pokemon)
    where
      foeMove = pickMove foe
      move = pickMove pokemon

instance Battle Water Fire
instance Battle Fire Water where
  battle = flip battle

instance Battle Grass Water
instance Battle Water Grass where
  battle = flip battle

instance Battle Fire Grass
instance Battle Grass Fire where
  battle = flip battle

printBattle :: String -> String -> String -> String -> String -> IO ()
printBattle pokemonOne moveOne pokemonTwo moveTwo winner = do
  putStrLn $ pokemonOne ++ " used " ++ moveOne
  putStrLn $ pokemonTwo ++ " used " ++ moveTwo
  putStrLn $ "Winner is: " ++ winner ++ "\n"
```
위 코드를 ghci 에서 불러들이고 다음 코드를 실행해 봅니다.

    > battle Squirtle Charmander
    Squirtle used Bubble
    Charmander used Ember
    Winner is: Squirtle

이제 위처럼 type annotation 없이도 잘 동작합니다.

한편, 지금까지 만든 코드에서는 battle 함수 호출시 첫번째 인자로 넘어간 Pokemon 이 항상 이기는 걸로 되어 있습니다. 이제 이 부분을 수정해서 인자 위치에 상관없이 어느 Pokemon 이 이길지를 코드에서 결정할 수 있도록 해 보겠습니다. 구체적으로 예를 들면 Fire 유형의 Pokemon 과 Water 유형의 Pokemon 이 싸우면 항상 Water 유형의 Pokemon 이 이기도록 하는 것입니다. 그러기 위해 Battel type class 에 Winner 라는 type 을 추가합니다. 그런데 Winner 는 새로운 type 이 아니라 기존에 Battle type class 의 두 개의 인자(둘 다 Pokemon type) 중 하나를 선택하는 것이므로 type synonym 입니다.
```haskell
class (Show (Winner a b), Pokemon a, Pokemon b) => Battle a b where
  type Winner a b :: *  -- 이것이 Associated type synonym 입니다.
  type Winner a b = a  -- default Implementation
  battle :: a -> b -> IO ()
  battle pokemon foe = do
    printBattle (show pokemon) (show move) (show foe) (show foeMove) (show winner)
    where
      foeMove = pickMove foe
      move = pickMove pokemon
      winner = pickWinner pokemon foe
  pickWinner:: a -> b -> Winner a b

instance Battle Water Fire where
  pickWinner a b = a
instance Battle Fire Water where
  type Winner Fire Water = Water
  pickWinner = flip pickWinner

instance Battle Grass Water where
  pickWinner a b = a
instance Battle Water Grass where
  type Winner Water Grass = Grass
  pickWinner = flip pickWinner

instance Battle Fire Grass where
  pickWinner a b = a
instance Battle Grass Fire where
  type Winner Grass Fire = Fire
  pickWinner = flip pickWinner
```
위 코드를 보면 Winner 를 결정하는 부분은 이제 instance 정의부에 있습니다. 위 코드를 ghci 에서 불러서 다음 코드를 실행해봅니다.

    > battle Wartortle Ivysaur
    Wartortle used WaterGun
    Ivysaur used VineWhip
    Winner is: Ivysaur

이제 battle 함수의 인자 순서에 무관하게 Grass 형 pokemon 이 Water 형 pokemon 을 이기고 있음을 알 수 있습니다.

이번에는 Family 개념에 대해 다루어보겠습니다. Memory 주소를 가르키는 pointer 는 미리 정해진 일정크기의 정수 단위로 정렬이 됩니다. 예를 들어 pointer 4 다음에는 8 이 오고 5 나 7은 유효한 pointer 주소가 아닙니다. 여기서는 pointer 주소를 type 수준에서 구분할 수 있도록 코드를 만들어 보겠습니다. 우선 다음처럼 몇 명 자연수를 type 으로 정의합니다.
```haskell
{-# LANGUAGE ScopedTypeVariables #-}
data Zero
data Succ n
type One = Succ Zero
type Two = Succ One
type Four = Succ (Succ Two)
type Six = Succ (Succ Four)
type Eight = Succ (Succ Six)

class Natural n where
  toInt:: n -> Int
instance Natural Zero where
  toInt _ = 0
instance (Natural n) => Natural (Succ n) where
  toInt _ = 1 + toInt (undefined:: n)
```
위 코드를 ghci 에서 불러들인 다음 다음을 실행해 봅니다.

    > toInt (undefined:: Two)
    2

참고로 Natural type class 의 type variable 'n' 은 phantom 입니다. 이제 Pointer 에 관한 type 을 만들겠습니다.
```haskell
{- 위의 코드에 이어서 작성합니다. -}
newtype Pointer n = MkPointer Int deriving Show
newtype Offset n = MkOffset Int deriving Show

multiple:: forall n. (Natural n) => Int -> Offset n
multiple i = MkOffset (i * toInt (undefined:: n))
```
위의 코드에서 Pointer n 이 뜻하는 것은 n-bytes 단위를 가진 Pointer 입니다. 즉, Pointer 8 이면 8 bytes 단위 Pointer 를 뜻합니다. Offset n 은 n-bytes 단위의 편차를 뜻합니다. 여기까지 작성한 코드를 ghci 에서 불러들인 다음 다음처럼 실행해봅시다.

    > let x = multiple 3::Offset Four
    > x
    MkOffset 12
    > :t x
    x :: Offset Four

x 에 12 bytes offset 을 뜻하는 값이 들어가 있고 그것의 type 이 Offset Four 임을 보여줍니다. 이제 Pointer 에 Offset 을 더하는 함수 add 를 만들겠습니다. add 의 결과로 나오는 Pointer type 은 Pointer 와 Offset 의 최대공약수가 될 것입니다. 즉, Pointer Eight 과 Offset Six 를 더한 결과의 type 은 Pointer Two 가 될 것입니다. 여기서 필요한 것이 GCD 라는 type function 으로 이를 통해 각 경우에 적합한 type 을 얻어옵니다. GCD 는 인자로 들어오는 것들의 type 에 따라 여러 개의 instance 를 가지며 이를family 로서 묶어서 같은 type 임을 뜻하는 것임을 명시합니다.

```haskell
{-# LANGUAGE TypeFamilies, ScopedTypeVariables #-}
{- 앞서 작성한 코드는 이 부분에 들어갑니다. -}
add:: Pointer m -> Offset n -> Pointer (GCD Zero m n)
add (MkPointer x) (MkOffset y) = MkPointer (x + y)

type family GCD d m n
type instance GCD d Zero Zero = d
type instance GCD d (Succ m) (Succ n) = GCD (Succ d) m n
type instance GCD Zero (Succ m) Zero = Succ m
type instance GCD (Succ d) (Succ m) Zero = GCD (Succ Zero) d m
type instance GCD Zero Zero (Succ n) = Succ n
type instance GCD (Succ d) Zero (Succ n) = GCD (Succ Zero) d n
```
지금까지 작성한 코드를 ghci 로 불러들인 다음에 다음 코드를 실행해봅니다.

    > let a = MkPointer 8::Pointer Eight
    > let b = MkOffset 6::Offset Six
    > let c = add a b
    > :t c
    c :: Pointer (Succ (Succ Zero))

이를 보면 add a b 의 결과값의 type 은 Pointer Two 로서 원래의 Pointer Eight 과는 정렬이 되지 않음을 type 수준에서 알 수 있습니다.

####TypeFamilyDependencies
이 확장은 GHC 8.0.1 부터 나오며 Type Family 코드에서 Functional dependency 를 표현할 수 있게 해 줍니다.
```haskell
```
#####Injective Type Families

####TypeInType
이 확장은 GHC 8.0.1 부터 나옵니다. 앞서 나왔던 PolyKinds 확장에서 한 걸음 더 나아간 것으로 이 확장을 쓰면 type 과 kind 가 같다고 선언하는 것이 됩니다. 원래 GHC 에서는 type 과 kind 를 별개의 것으로 구분한

참고로 이 확장에서 다루고 있는 kind system 에 대한 논문은 [System FC with Expilicit Kind Equality](http://www.seas.upenn.edu/~sweirich/papers/fckinds.pdf) 입니다.

####TypeOperators
이 확장을 사용하면 (+) 와 같은 operator 들을 다음 처럼 type constructor 로서 사용할 수 있습니다.
```haskell
{-# LANGUAGE TypeOperators #-}
data a + b = Plus a b
type Foo = Int + Bool
```

    > let a = Plus (9::Int) True
    > :t a
    a :: Int + Bool

####LiberalTypeSynonyms
이 확장을 쓰면 Haskell 에서 type synonym 을 정의할 때 가하는 여러 가지 제약을 완화할 수 있습니다. 예를 들어 type synonym 은 partial application 이 안 되게 되어 있는데 이 확장을 쓰면 가능합니다. 아래 코드를 보면 Const 나 Id 는 모두 받아야 할 인자가 하나씩 모자란 상태로 myFunc의 type signature에서 쓰이고 있습니다.
```haskell
{-# LANGUAGE LiberalTypeSynonyms #-}
import Data.Char(ord)

type Const a b = a
type Id a = a
type NatApp f g i = f i -> g i

myFunc :: NatApp Id (Const Int) Char
--     ~  Id Char -> Const Int Char
--     ~  Char    -> Int
myFunc = ord
```
또한, forall 예약어를 type synonym 에 사용할 수 있습니다.
```haskell
{-# LANGUAGE LiberalTypeSynonyms, RankNTypes #-}
type Discard a = forall b. Show b => a -> b -> (a, String)

f :: Discard a
f x y = (x, show y)

g :: Discard Int -> (Int,String)    -- A rank-2 type
g f = f 3 True
```
이렇게 유연한 type synonym 을 사용할 수 있는 이유는 LiberalTypeSynonyms 확장을 쓰면 type synonym 을 확장한 이후에야 type check 이 이루어지기 때문입니다.

####DefaultSignatures
이 확장은 Generic Programming 과 관련되어 있는 확장으로 특정 instance 에 대한 default 구현을 허용해줍니다. 즉, 아래 코드에서 enum 함수의 type 은 [a] 이지만 만약에 Enum 의 instance 중에 하나가 instance (Generic a, GEnum (Rep a)) => Enum a 로 되어 있으면 해당 instance 는 default 예약어가 붙어 있는 enum 함수를 사용하게 되는 것입니다. 이 default 예약어가 붙어 있는 enum 함수의 type signature 가 [a] 가 아닐 수 있게 하는 것이 이 확장의 기능입니다.
```haskell
{-# LANGUAGE DefaultSignatures #-}
import GHC.Generics

class Enum a where
  enum :: [a]
  default enum :: (Generic a, GEnum (Rep a)) => [a]
  enum = map to genum
```
####ImplicitParams
이 확장은 함수 인자를 묵시적으로 지정할 수 있게 합니다. 좀 더 구체적으로는 함수의 특정 인자를 callee 의 입장에서 binding 하지 않고 caller 의 입장에서 binding 하는 dynamic binding 에 관한 것입니다. 아래 코드에서 sort 함수 type signature 의 constraint 부분에서 물음표가 앞에 붙어 있는 ?cmp 부분이 바로 implicit parameter 입니다.
```haskell
{-# LANGUAGE ImplicitParam #-}
import Data.List (sortBy)

sort :: (?cmp :: a -> a -> Ordering) => [a] -> [a]
sort = sortBy ?cmp
least xs = head (sort xs)
```
위 코드를 ghci 에서 불러들인 다음 코드를 실행해보겠습니다.

    >:set -XImplicitParams
    > let ?cmp = compare in least [12,1,9,3]
    1

이처럼 ?cmp 인자를 callee(least 함수) 에서 직접 넘기는 것이 아니라 caller(위의 let 구문) 에서 넘기고 있습니다.

####ConstraintKinds
constraints (=> 기호 왼쪽에 오는 부분) 는 매우 제한적인 문법을 가지고 있는데, 다음 세 가지의 경우만 허용 합니다.

* 첫째, Show a 와 같은 class constraints.
* 둘째, 앞서 다루었던 ?x::Int 와 같은 Implicit parameter.
* 셋째, Type Family 에서 등장하는 a ~ Int 와 같은 Equality constraints.

그런데 ConstraintKinds 확장을 쓰면 constraints 로 좀 더 다양한 것을 쓸 수 있습니다. 예를 들어 다음처럼 tuple 을 쓸 수 있습니다.

```haskell
{-# LANGUAGE ConstraintKinds #-}
type Stringy a = (Read a, Show a)
foo :: Stringy a => a -> (String, String -> a)
foo x = (show x, read)
```

## 첫날 세번째 100분
- ApplicativeDo
- StandaloneDeriving
- Typed Holes
- Monad Transformers
- REPA(REgular PArallel arrays)

####ApplicativeDo
Monad 의 경우 do notation 을 사용하여 bind 동작을 좀 더 이해하기 쉬운 형태로 코드를 작성할 수 있습니다. ApplicativeDo 확장을 쓰면 do notation 을 Applicative 의 경우에도 쓸 수 있습니다. 다음 코드에서 ZipList type 은 Applicative 이지만 Monad 는 아닙니다. 따라서 do notation 을 사용할 수 없습니다.
```haskell
import Control.Applicative
pp = (*) <$> ZipList [1,2,3] <*> ZipList [7,8,9] -- ZipList {getZipList = [7,16,27]}
```
하지만 ApplicativeDo 확장을 쓰면 위 코드를 다음처럼 do notation 을 이용하여 좀 더 보기 쉬운 형태로 작성할 수 있습니다.
```haskell
{-# LANGUAGE ApplicativeDo #-}
import Control.Applicative
pp = do
  a <- ZipList [1,2,3]
  b <- ZipList [7,8,9]
  return (a*b)
```

####StandaloneDeriving
다음처럼 자료형 만들 때 deriving 을 함께 하지 않고 별도로 하는 것을 말합니다.
```haskell
{-# LANGUAGE StandaloneDeriving #-}
data Foo a = Bar a | Baz String
deriving instance Eq a => Eq (Foo a)
```
그렇다면 이렇게 하는 것은 어떤 장점이 있을까요? 우선 다음처럼 특정 instance 만 콕 집어서 deriving 할 수 있습니다.
```haskell
{-# LANGUAGE StandaloneDeriving #-}
data Foo a = Bar a | Baz String

deriving instance Eq a => Eq (Foo [a])
deriving instance Eq a => Eq (Foo (Maybe a))
```
위처럼 했을 때 (Foo [a]) 와 (Foo (Maybe a)) 는 Eq 의 instance 이지만 그 외의 것, 가령 (Foo (Int,Bool)) 는 Eq 의 instance 가 아닙니다.

또, 한 가지 이점은 일반적인 deriving 을 할 수 없는 GADTs 나 그 밖의 특이한 data type 들에 대해서도 deriving 을 할 수 있다는 것입니다. 예를 들어 다음과 같은 GADTs 코드에는 derivivg (Show) 같은 코드를 바로 붙일 수가 없습니다.
```haskell
data T a where
  T1 :: T Int
  T2 :: T Bool
```
따라서 이 때는 StandaloneDeriving 확장을 통해 다음처럼 해야 합니다.
```haskell
{-# LANGUAGE StandaloneDeriving #-}
data T a where
  T1 :: T Int
  T2 :: T Bool

deriving instance Show (T a)
```

####Typed Holes

####Monad Transformers

####REPA(REgular PArallel arrays)

## 둘째날 첫 100분
- DWARF based debugging
- Template Haskell with Quasiquoting

## 둘째날 두번째 100분
- Dependent Types

## 둘째날 세번째 100분
- QuickCheck
  * shrinking

####QuickCheck
QuickCheck 은 '속성 기반 테스팅'(Property-based testing) 을 위한 라이브러리 입니다. 먼저 사용법을 살펴보겠습니다. 다음 revList 함수를 제대로 작성했는지 검증하는 것을 생각해봅시다.
```haskell
revList:: [a] -> [a]
revList [] = []
revList (x:xs) = revList xs ++ [x]
```
revList 함수의 속성 중 하나는 revList 함수를 두 번 연속 수행하면 원래의 입력과 같다는 것입니다. 이를 다음처럼 함수로서 표현할 수 있습니다. 함수 이름 앞에 붙은 prop 은 property 를 뜻하는 것으로 이처럼 QuickCheck 라이브러리 사용시 속성을 뜻하는 함수 이름을 지을 때 관행처럼 붙입니다.
```haskell
prop_ReverseTwiceIsSame::[a] -> Bool
prop_ReverseTwiceIsSame xs = revList (revList xs) == xs
```
이제 ghci 에서 위 코드를 불러들인 다음 QuickCheck 의 기능을 사용해서 검증해보겠습니다.

    > import Test.QuickCheck
    > quickCheck prop_ReverseTwiceIsSame
    +++ OK, passed 100 tests.

무슨 일이 일어난 것일까요? 위에서 사용한 quickCheck 함수 대신 verboseCheck 함수를 써서 무슨 일이 일어났는지 짐작해보겠습니다.

    > verboseCheck prop_ReverseTwiceIsSame
    Passed:
    []
    Passed:
    []
    Passed:
    [2,2]
    Passed:
    [3,-3,2]
    ...
    +++ OK, passed 100 tests.

위 결과에서 보이듯이 QuickCheck 은 Int 형 요소를 가진 List 를 임의로 100 개 만들어서 이것들이 prop\_ReverseTwiceIsSame 함수를 만족하는지 검사합니다. 사람이 직접 Test input 을 만들지 않아도 되는 것입니다. 이번에는 다른 속성을 가지고 검증해 보겠습니다.
```haskell
prop_ReverseIsSame::[Int] -> Bool
prop_ReverseIsSame xs = revList xs == xs
```

    > quickCheck prop_ReverseIsSame
    *** Failed! Falsifiable (after 3 tests and 3 shrinks):
    [0,1]

위처럼 QuickCheck 은 prop\_ReverseIsSame 함수가 거짓값이 나올 때까지 임의의 Test input 을 만들어서 검증합니다. 위에서는 세번째 테스트만에 prop\_ReverseIsSame 속성이 거짓이 되는 입력값이 나왔으며 해당 입력값은 [0,1] 이라고 알려주고 있습니다.

만약, 100 개의 Test 입력이 부족하다고 느낄 때는 다음처럼 quickCheckWith 를 써서 직접 Test 입력 갯수를 지정해줄 수 있습니다.

    > quickCheckWith stdArgs {maxSuccess = 500} prop_ReverseTwiceIsSame
    +++ OK, passed 500 tests.

또한, 무작위로 만드는 List 각각의 최대 길이를 직접 지정할 수도 있습니다.

    > verboseCheckWith stdArgs {maxSize = 6} prop_ReverseTwiceIsSame
    ...

이렇게 하면 길이가 6 이하인 임의의 List 들만 Test 입력으로 만들어집니다.

이처럼 QuickCheck 은 Test 입력을 무작위로 만들어서 검증하는 방법입니다. 그렇다면 QuickCheck 이 무작위로 Test 입력을 만드는 것은 어떻게 이루어지는 것일까요? QuickCheck 은 어떤 무작위로 만들어진 값 a 를 뜻하는 것으로 이것의 type 을 Gen a 로서 정의합니다. 그리고 이것의 기본 생성함수(generator 라고 합니다)를 arbitrary 라고 정하고 arbitrary 함수를 가진 type 들을 Arbitrary 라는 typeclass 로 묶기로 합니다.
```haskell
data Gen a

class Arbitrary a where
  arbitrary:: Gen a
```
이러한 생성자(generator)는 generate 라는 함수에 넘겨서 실제로 값을 만들어내도록 합니다.
```haskell
generate::Gen a -> IO a
```

    > import Test.QuickCheck
    > generate arbitrary::IO Int
    -4
    > generate arbitrary::IO (Maybe Int)
    Just (-2)
    > generate arbitrary::IO [Maybe Bool]
    [Just False,Just False,Just False,Just True,Nothing,Just True]
    > generate arbitrary::IO (Int, Char)
    (5,'`')
    > generate arbitrary::IO (Either Int Double)
    Left 25

또한 QuickCheck 은 생성자를 편하게 쓸 수 있도록 몇 가지 조합함수를 제공합니다. 가령 특정 범위에 있는 것들 중에서 무작위로 값을 뽑아내는 용도로 choose 를 제공합니다. 이를 이용해서 주사위 굴리기를 흉내내보겠습니다.
```haskell
choose:: Random a => (a, a) -> Gen a
```

    > import Test.QuickCheck
    > let dice::Gen Int; dice = choose (1,6)
    > generate dice
    5

만약에 특정 범위의 값을 좀 더 높은 비율로 만들고 싶을 때는 frequency 를 이용합니다. 아래에서는 3:1 비율로 0~50 사이의 임의의 수와 51~100 사이의 임의의 수를 만듭니다.

    > let biased::Gen Int; biased = frequency [(3, choose (0,50)), (1, choose (51,100))]

Bool 에 대한 무작위 값을 내놓을 때도 choose 조합함수를 이용합니다. 다음 코드는 QuickCheck 이 정의하고 있는 Bool type 에 대한 Arbitrary instance 입니다.
```haskell
instance Arbitrary Bool where
  arbitrary = choose (False,True)
```

무작위로 List 를 만드는 생성자를 만들 때도 choose 를 사용합니다. 먼저 List 의 경우 길이라는 속성이 추가로 필요합니다. 무작위로 만드는 List 각각의 최대 길이를 지정해주어야 합니다. 그렇지 않으면 무한대 길이의 List 를 만들 수도 있으니까요. 그래서 다음의 sized 라는 것이 필요합니다.
```haskell
sized:: (Int -> Gen a) -> Gen a
```
이제 이를 이용해서 arbitraryList 라는 List 생성자를 만들어보겠습니다.
```haskell
arbitraryList:: Arbitrary a => Gen [a]
arbitraryList =
  sized $
    \n -> do
      k <- choose (0, n)
      sequence [arbitrary | _ <- [1..k]]
```
위의 코드에서 n 변수는 quickQueck 을 호출할 때 QuickCheck 라이브러리 내부적으로 정한 기본값이 들어갑니다. 앞서 나왔던 maxSize 라는 환경값을 설정해주는 것이 바로 이 n 변수값을 설정하는 것입니다.

이제 위 생성자를 통해 무작위로 List 를 만들어봅니다.

    > generate arbitraryList:: IO [Int]
    [-14,27,27,13,30,17,7,8,-1,26,9]
    > generate arbitraryList:: IO [Bool]
    [False,False]

위에서 만든 arbitraryList 의 코드는 Test.QuickCheck 라이브러리에서 List 를 Arbitrary instance 로 만들 때 사용하는 코드와 같습니다.

다음으로 우리가 직접 만든 자료형에 대한 생성자를 만들어보겠습니다. 다음 Rose tree 를 예로 들겠습니다.
```haskell
data Tree a = Tree a [Tree a] deriving (Show)
```
앞서 List 에 대한 generator 를 만들었던 것과 비슷하게 다음처럼 만들 수 있습니다.
```haskell
instance Arbitrary a => Arbitrary (Tree a) where
  arbitrary = do
    t <- arbitrary
    ts <- arbitrary
    return (Tree t ts)
```
그런데 이렇게 할 경우에는 List 의 경우와 마찬가지로 무한대 크기의 Tree 를 만들 수도 있게 되어 문제가 됩니다. 따라서 여기서도 sized 를 사용하여 다음처럼 작성해야 합니다.
```haskell
instance Arbitrary a => Arbitrary (Tree a) where
  arbitrary = sized arbitrarySizedTree

arbitrarySizedTree:: Arbitrary a => Int -> Gen (Tree a)
arbitrarySizedTree m = do
  t <- arbitrary
  n <- choose (0, m `div` 2)
  ts <- vectorOf n (arbitrarySizedTree (m `div` 4))
  return (Tree t ts)
```
위 코드에서 vectorOf 는 choose 와 마찬가지로 편의를 위해 제공하는 조합함수로서 다음과 같은 type 을 가집니다.
```haskell
vectorOf:: Int -> Gen a -> Gen [a]
```
그리고 m 을 2 와 4 로 나누는 코드가 있는 것은 Tree 크기가 지나치게 커지는 것을 막으려는 목적이며 다른 뜻이 있는 것은 아닙니다. 이제 이 생성자를 이용하여 무작위로 Tree 를 만들어보겠습니다.

    > generate arbitrary::IO (Tree Int)
    Tree (-6) [Tree 30 [Tree (-30) [],Tree 0 [],Tree 28 []],Tree 14 []]
    > generate arbitrary:: IO (Tree Char)
    Tree ';' [Tree '\159' [Tree '0' [],Tree 'P' []]]

이제 Tree 의 속성을 검증해보겠습니다. 우리가 만든 Rose Tree 의 속성 중 하나는 Node의 수가 Edge 의 수보다 하나 더 많다는 것입니다. 따라서 다음과 같은 속성을 만들어볼 수 있습니다.
```haskell
prop_OneMoreNodeThanEdges:: Tree a -> Bool
prop_OneMoreNodeThanEdges tree = nodes tree == edges tree + 1

nodes:: Tree a -> Int
nodes (Tree t []) = 1
nodes (Tree t ts) = 1 + sum(map nodes ts)

edges:: Tree a -> Int
edges (Tree t []) = 0
edges (Tree t ts) = length ts + sum(map edges ts)
```
이제 prop\_OneMoreNodeThanEdges 속성을 quickCheck 으로 테스트해 보면 다음처럼 통과함을 볼 수 있습니다.

    > quickCheck prop_OneMoreNodeThanEdges
    +++ OK, passed 100 tests.

이제 Quick Check 의 내부를 좀 더 들여다보겠습니다. 먼저 quickCheck 함수의 type 을 보겠습니다.
```haskell
-- The set of types that can be tested
class Testable prop

quickCheck:: Testable prop => prop -> IO ()
```
이를 보면 quickCheck 이 받는 인자는 Testable 의 instance 이어야 합니다. 그런데 방금 검증한 prop\_OneMoreNodeThanEdges 속성은 Testable 의 instance 가 아니었습니다. 다시 확인해 보겠습니다.

    > :t quickCheck
    quickCheck :: Testable prop => prop -> IO ()
    > :t prop_OneMoreNodeThanEdges
    prop_OneMoreNodeThanEdges :: Tree a -> Bool
    > quickCheck prop_OneMoreNodeThanEdges
    +++ OK, passed 100 tests.

type 을 보면 Tree a -> Bool 일뿐 Testable 의 instance 는 아닙니다. 그런데 어떻게 quickCheck 이 동작하는 것일까요? 그건 바로 다음 두 개의 Testable instance 를 통해서입니다.
```haskell
-- Satisfied by the result type
instance Testable Bool

-- Satisfied by the argument and result
instance (Arbitrary a, Show a, Testable prop) => Testable (a -> prop)
```
quickCheck 함수에 우리가 검증하고자 하는 함수(속성을 뜻하는)를 인자로서 넘기는데, 이 때 인자로 넘어가는 함수자체의 인자는 Arbitrary 와 Show 의 instance 이어야 하고, 함수의 결과는 Testable 의 instance 이어야 합니다. 먼저 함수의 결과 type 에 대해 살펴보면, 우리가 속성을 표현하기 위해 만드는 함수들은 모두 결과가 Bool type 인데 Bool type 은 위 코드에서 나오듯이 Testable 의 instance 로 QuickCheck 에서 정의하고 있으므로 조건을 만족합니다. 다음으로 함수의 인자들은 앞서 봤듯이 Arbitrary 의 instance 로 만드는 작업을 해주었고 deriving 을 통해 Show 의 instance 로 만들기도 했으므로 이 역시 조건을 만족합니다. Tree 의 다음 코드 부분을 보면 이를 확인할 수 있습니다.
```haskell
data Tree a = Tree a [Tree a] deriving (Show, ..)
instance Arbitrary a => Arbitrary (Tree a) where ...
```
이렇게 quickCheck 함수에 위에서 언급한 조건을 만족하는 함수를 인자로서 넘기면 quickCheck 은 모든 필요한 type 의 임의의 값을 무작위로 자동으로 만들어냅니다. 그리고 나서 그 만들어진 값으로 우리가 만든 test(속성을 나타내는 함수)를 돌립니다. 그리고 나서 모든 경우에 test 를 통과하는지 확인합니다.

예제를 하나 더 해 보겠습니다. Char type 은 Unicode 를 나타냅니다.

    > let a::Char; a = '\x10FFFF'
    > a
    '\1114111'

Char 하나(code point)를 UTF-16 으로 인코딩하는 함수를 작성하고 이를 검증해 보겠습니다. 참고로 UTF-16 인코딩은 Unicode 문자 하나를 하나 또는 두 개의 16비트 code unit 으로 바꿉니다. 0x10000 이하의 code point 는 1개의 16비트 code unit 으로, 0x10000 이상의 code point 는 2개의 16비트 code unit 으로.
```haskell
import Data.Word (Word16)
import Data.Char (ord)
import Data.Bits ((.&.), shiftR)

encodeUTF16:: Char -> [Word16]
encodeUTF16 c
  | w < 0x10000 = [fromIntegral w] -- single code unit
  | otherwise = [fromIntegral a, fromIntegral b] -- two code unit
   where w = ord c
         a = ((w - 0x10000) `shiftR` 10) + 0xD800
         b = (w .&. 0x3FF) + 0xDC00
```
그리고 나서 이 함수의 속성을 하나 생각해보겠습니다. 이번에는 잘못된 속성을 하나 만들어서 테스트해보겠습니다.
```haskell
prop_encodeOne c = length (encodeUTF16 c) == 1
```
인코딩된 결과의 길이는 1 또는 2 니까 위의 속성은 테스트를 통과하지 못해야 합니다. 그런데 실제 해 보면 다음처럼 통과합니다.

    > import Test.QuickCheck
    > quickCheck prop_encodeOne
    +++ OK, passed 100 tests.

이는 QuickCheck 에서 Char type 의 Arbitrary instance 가 ASCII 값만 만들게 되어 있기 때문입니다. sample 함수를 써서 간단히 이를 확인해 볼 수 있습니다. sample 함수는 말 그대로 해당 type 의 임의의 값을 몇개 무작위로 뽑아서 예시로 보여줍니다.

    > sample (arbitrary::Gen Char)
    '\230'
    'E'
    ...

다음 코드는 QuickCheck 소스에서 Char type 의 Arbitrary instance 정의 부분입니다.
```haskell
instance Arbitrary Char where
  arbitrary = chr `fmap` oneof [choose (0,127), choose (0,255)]
```
따라서 위에서 실패할 것으로 기대했던 속성이 성공했던 것입니다. 참고로 이렇게 Char type 의 무작위 생성값이 ASCII 값만 나오게 되어 있는 것은 일부러 그렇게 한 것입니다.

따라서 원래 생각했던 바를 검증하려면 직접 별도의 type 을 정의해야 합니다. 다음 코드처럼.
```haskell
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
import Test.QuickCheck
import System.Random

newtype BigChar = Big Char deriving (Eq, Show, Random)

instance Arbitrary BigChar where
  arbitrary = choose (Big '0',Big '\x10FFFF')

prop_encodeOne2 (Big c) = length (encodeUTF16 c) == 1
```
그리고 나서 다시 quickCheck 을 돌려보면 다음처럼 테스트를 통과하지 못함을 확인할 수 있으며 어떤 입력에 대하여 실패하는지도 알려줍니다.

    > quickCheck prop_encodeOne2
    *** Failed! Falsifiable (after 1 test):
    Big '\89420'

#####shrinking
무작위 입력값 발생 기능과 함께 QuickCheck 이 제공하는 또 다른 중요기능은 shrinking 입니다. 어떤 실패하는 입력을 찾았을 때 그 입력값의 크기가 매우 크다면 디버깅하기 불편할 것입니다. 예를 들어 Tree 에 대하여 테스트할 때 실패하는 입력값이 크기가 1000 짜리의 큰 Tree 라면 그것의 어떤 부분에서 잘못된 것이 있는지 디버깅하기 쉽지 않을 것입니다. Shrinking 기능은 어떤 속성에 대하여 반례를 찾았을 때 찾은 반례보다 조금씩 더 작은 크기의 반례를 찾아가는 기능입니다. 최종적으로 가장 작은 크기의 반례를 찾도록 해줍니다. 앞서 만들었던 BigChar 의 Arbitrary instance 에 shrinking 기능을 추가해 보겠습니다. 이는 shrink 함수를 정의해주면 됩니다. shrink 함수가 하는 일은 반례 하나를 받아서 그것보다 작은 크기의 여러 개의 반례들의 목록을 만드는 것입니다.
```haskell
import Data.Char (ord,chr)

instance Arbitrary BigChar where
  arbitrary = choose (Big '0',Big '\x10FFFF')
  shrink (Big c) = map Big (shrinkChar c)

shrinkChar c = map (chr.floor) (lst c)
  where lst c = sequence [(*0.5), (*0.75), (*0.875)] (fromIntegral.ord $ c)
```
위 코드에서 BigChar 에 대한 shrink 함수는 code point 하나를 받아서 그것보다 일정 크기로 작은 code point 들을 세 개 만들고 있습니다. 그러면 quickCheck 이 이렇게 만든 세 개의 code point 에 대해서 테스트를 진행하여 테스트가 또 다시 실패하면 다시 해당 code point 에 대하여 shrinking 을 수행합니다. 이러한 과정을 반복하여 거치면서 테스트가 실패하는 좀 더 작은 code point 를 찾는 것입니다. Shrinking 기능을 추가하고 나서 quickCheck 을 돌리면 다음과 같이 shrinking 이 수행된 결과를 볼 수 있습니다.

    > quickCheck prop_encodeOne2
    *** Failed! Falsifiable (after 1 test and 5 shrinks):
    Big '\70119'

shrink 함수는 List 에 대해서는 어떻게 동작할까요? 시험해보겠습니다.

    > shrink [2,3]
    [[],[3],[2],[0,3],[1,3],[2,0],[2,2]]
    > shrink [8]
    [[],[0],[4],[6],[7]]

앞서 만들었던 Rose Tree 의 Arbitrary instance 에도 shrinking 기능을 넣어보겠습니다. 다음처럼 할 수 있습니다.
```haskell
import Data.List (subsequences)

instance Arbitrary a => Arbitrary (Tree a) where
  arbitrary = sized arbitrarySizedTree
  shrink (Tree t ts) =
      [Tree t' ts| t' <- shrink t] ++
      [t' | t' <- ts] ++
      init [Tree t ts'| ts' <- subsequences ts]
```
마지막으로 유용한 몇 가지 기능을 더 살펴보겠습니다. 먼저 다음 코드를 보십시요.
```haskell
import Test.QuickCheck((==>), suchThat)

prop_encodeOne3 = do
  c <- choose ('\0', '\xFFFF')
  return $ length (encodeChar c) == 1

prop_encodeOne4 (Big c) =
  (c < '\x10000') ==> length (encodeChar c) == 1

prop_encodeOne5 = do
  Big c <- arbitrary `suchThat` (< Big '\x10000')
  return $ length (encodeChar c) == 1
```
위 코드에서 (==>) 와 suchThat 은 무작위 생성값을 거르는 역할을 합니다. 테스트입력을 만들어낼 때 prop\_encodeOne3 처럼 처음부터 적합한 값을 만드는 방법도 있지만 prop\_encodeOne4 & 5 처럼 (==>) 와suchThat 을 이용하여 적합한 값만 걸러낼 수도 있습니다. 이는 선택의 문제이지만 보통 처음부터 적합한 것을 만드는 것이 좀 더 효율적입니다. 그리고 적합한 값을 걸러내는 방식을 택할 경우 QuickCheck 이 우리가 원하는 만큼 충분히 테스트하지 못하게 됩니다. 따라서 되도록 처음부터 적합한 값을 만들도록 하는 것을 권장합니다.

List 에 요소를 하나 삽입하는 함수를 검증하는 코드를 살펴봅시다. 참고로 아래 코드에서 types = x::Int 부분의 types 는 예약어가 아니라 그냥 변수 이름으로서 실제는 사용하지 않는 dummy variable 이며 오로지 x::Int 라는 type signature 를 주기 위해서만 존재합니다. 예약어가 아니므로 types 라는 이름 대신 abc 같은 이름을 써도 상관없습니다.
```haskell
ascending::Ord a => [a] -> Bool
ascending (x:x':xs) = x <= x' && ascending (x':xs)
ascending _ = True

insert:: Ord a => a -> [a] -> [a]
insert x [] = [x]
insert x (y:ys)
  | x < y = x:y:ys
  | otherwise = y:insert x ys

prop_ins_ord x xs = ascending xs ==> ascending (insert x xs) where types = x::Int
```
이를 quickCheck 함수로 실행해 보면 다음과 같은 결과를 볼 수 있습니다.

    > quickCheck prop_ins_ord
    *** Gave up! Passed only 69 tests.

이것은 앞서 말했듯이 (==>) 을 써서 적합한 입력값만 받게 했을 때 나오는 결과로 QuickCheck 이 69 개의 test 만 수행했고 나머지 31 개는 (==>) 에 걸린 조건에 맞지 않아서 수행하지 못했다는 것을 알려줍니다.

그렇다면 실제로 수행된 테스트들은 어떤 것들이었는지 확인해보고 싶을 수 있습니다. 이 때는 일종의 통계기능을 사용할 수 있습니다. collect 함수를 쓰면 확인하고자 하는 것(여기서는 입력값의 길이)을 기준으로 대한 입력값의 분포를 알수 있습니다.
```haskell
prop_ins_ord2 x xs = collect (length xs) $ prop_ins_ord x xs
```
    > quickCheck prop_ins_ord2
    *** Gave up! Passed only 65 tests:
    35% 1
    33% 0
    18% 2
     7% 3
     3% 5
     1% 6

classify 함수를 이용하면 딱 확인하고자 하는 것만 짧게 알려줍니다.
```haskell
prop_ins_ord3 x xs = clasify (length xs < 2) "too small!" $ prop_ins_ord x xs
```
    > quickCheck prop_ins_ord3
    *** Gave up! Passed only 82 tests (85% too small!).

이처럼 적합한 값만 거르도록 했을 때는 테스트 수행이 생각보다 만족스럽지 못할 수 있습니다. 그렇다면 처음부터 적합한 값을 만들도록 하는 방법은 뭐가 있을까요? Arbitrary instance 에 있는 기본 생성자를 쓰지 않고 생성자를 따로 명시하면 되겠지요. 이를 위해 forAll 조합함수가 있습니다.

forAll 조합함수는 forAll "generator" $ \pattern -> "property to test" 꼴로 사용합니다. 다음 Fibonacci 수 생성기를 검증하는 코드에서 인덱스를 500 이하 양의 정수로 제한하도록 smallNonNegativeIntegers 라는 생성자를 명시적으로 넘기고 있습니다.
```haskell
fibs = 0 : 1 : zipWith (+) fibs (tail fibs)

smallNonNegativeIntegers = choose (0, 500)

prop_Fibonacci =
  forAll smallNonNegativeIntegers $ \n ->
    let x = fibs !! (n)
        y = fibs !! (n+1)
        z = fibs !! (n+2)
    in x + y == z
```
좀 전에 List 삽입하는 함수 검증도 이를 이용하면 다음처럼 좀 더 바람직하게 할 수 있습니다. 아래코드에서 orderedList 생성자는 QuickCheck 라이브러리에서 제공하는 생성자입니다.
```haskell
prop_ins_ord4 x =
  forAll orderedList $ \xs ->
    collect (length xs) $ ascending (insert x xs)
      where types = x:: Int
```

    > quickCheck prop_ins_ord4
    +++ OK, passed 100 tests:
    5% 6
    5% 0
    4% 4
    4% 2
    ...

## 더 읽을 거리
#### Zipper
#### Finger trees
#### Hash Array Mapped Trie (HAMT)
#### Stream fusion

## License
Eclipse Public License
