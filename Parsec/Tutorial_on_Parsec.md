###Parsec 배우기
Parsec 라이브러리를 쓰면 파싱(parsing) 작업을 매우 효율적으로 처리할 수 있습니다. 먼저 파싱이란 무엇인지에 대해 생각해 봅시다.

파싱을 아주 단순하게 표현해 보면, 임의의 문자열을 입력으로 받아서 해당 문자열에서 특정 pattern 을 찾아내는 것이라고 할 수 있습니다. 조금 더 설명을 덧붙여 보면 입력 문자열에서 특정 pattern 을 찾을 수 있을 만큼 찾고나서 찾은 pattern 과 분석하지 못하고 남은 문자열을 함께 돌려주는 것이라고 할 수 있습니다. 이 설명을 type 으로 옮겨보면 다음과 같습니다.
```haskell
type Parser s a = s -> (a, s)
```
여기서 s 는 입력을, a 는 찾고자 하는 pattern 을 뜻합니다. 그런데 하나 더 고려해야 할 게 있습니다. 바로 파싱이 항상 성공하는건 아니라는 것입니다. 따라서 성공할 수도 실패할 수도 있는 값을 뜻하는 Maybe 를 씌워야 합니다. 이제 다음처럼 type 을 바꿉니다.
```haskell
type Parser s a = s -> Maybe (a, s)
```
이제 아주 간단한 Parser 하나를 만들어 보겠습니다. 입력 문자열이 특정 pattern 으로 시작하는지 검사하는 Parser 입니다.
```haskell
import Data.List (stripPrefix)

type Parser s a = s -> Maybe (a,s)

stringBegins :: String -> Parser String String
stringBegins pattern = \input ->
  case stripPrefix pattern input of
    Nothing   -> Nothing
    Just rest -> Just (pattern, rest)
```

    > stringBegins "abc" "abcd"
    Just ("abc","d")
    > stringBegins "abc" "axyzbc"
    Nothing

이번에는 문자열 처음에서 숫자부분을 검사하는 Parser 를 만들겠습니다. 위의 파서만큼 간단합니다.
```haskell
import Data.List (span)
import Data.Char (isDigit)

number:: Parser String Int
number input =
  if h == "" then Nothing else Just (read h::Int, t)
  where (h, t) = span isDigit input
```

    > number "98abc"
    Just (98,"abc")
    > number "abc98"
    Nothing

이렇게 만든 stringBegins, number 두 개의 파서를 가지고 다음처럼 동작하는 version 이라는 아주 간단한 파서를 만들어봅니다.

    > version "version 8.0.1" -- 이 함수의 결과는 (8,0,1) 이면 될 것 같습니다.

단순하게는 다음처럼 만들 수 있습니다.
```haskell
version1 i0 =
  case stringBegins "version " i0 of
    Nothing -> Nothing
    Just (_,i1) ->
      case number i1 of
        Nothing -> Nothing
        Just (major,i2) ->
          case stringBegins "." i2 of
            Nothing -> Nothing
            Just (_,i3) ->
              case number i3 of
                Nothing -> Nothing
                Just (minor,i4) ->
                  case stringBegins "." i4 of
                    Nothing -> Nothing
                    Just (_,i5) ->
                      case number i5 of
                        Nothing -> Nothing
                        Just (revision,i6) -> Just((major,minor,revision),i6)
```
이제 좀 더 나은 방법으로 다시 작성하려보니 위 코드는 다음과 같은 패턴의 반복임을 알 수 있습니다.

Parser 에 입력을 준다 → Parser 는 입력을 분석해서 찾으려는 패턴을 찾는다 → 찾으려는 패턴을 찾아서 분석이 성공하면 입력에서 남은 부분을 다음 Parser 로 넘긴다

이 패턴을 함수로 만들어보면 다음과 같습니다. 함수 이름은 andThen 으로 하겠습니다.
```haskell
andThen :: Parser s a -> (a -> Parser s b) -> Parser s b
andThen parse next = \input ->
  case parse input of
    Nothing          -> Nothing
    Just (a, input') -> next a input'
```
이제 이 함수를 이용해서 version 함수를 다시 써보면 다음과 같습니다.
```haskell
version2 =
  stringBegins "version " `andThen` \_ ->
  number `andThen` \major ->
  stringBegins "." `andThen` \_ ->
  number `andThen` \minor ->
  stringBegins "." `andThen` \_ ->
  number `andThen` \revision ->
  {- ... (major, minor, revision) 결과를 돌려주는 코드 ...-}
```
위 코드의 마지막 줄에서 결과를 돌려줄 때는 Parser 에 담아서 돌려주므로 이를 위한 함수도 따로 만듭니다. 함수 이름은 pack 으로 하겠습니다.
```haskell
pack :: a -> Parser s a
pack a = \input -> Just (a, input)
```
이제 코드를 붙여보면 다음과 같습니다.
```haskell
version2 =
  stringBegins "version " `andThen` \_ ->
  number `andThen` \major ->
  stringBegins "." `andThen` \_ ->
  number `andThen` \minor ->
  stringBegins "." `andThen` \_ ->
  number `andThen` \revision ->
  pack (major, minor, revision)
```
여기서 만든 andThen 함수와 pack 함수가 어디 다른 곳에서 보았던 것과 비슷하다고 느낀다면 그 느낌이 맞습니다. 다음 코드를 보십시요.
```haskell
-- andThen 과 (>>=) 이 같은 꼴입니다.
andThen :: Parser s a -> (a -> Parser s b) -> Parser s b
(>>=)   :: Monad m =>
           m        a -> (a -> m        b) -> m        b

-- pack 과 return 이 같은 꼴입니다.
pack   ::            a -> Parser s a
return :: Monad m => a -> m        a
```
그렇습니다. Parser 는 사실 Monad 입니다. 그런데 앞서 우리가 Parser 를 type synonym 으로 정의했기 때문에 현재 상태에서는 이를 Monad 의 Instance 로 만들 수 없습니다. 왜냐하면 type synonym 은 어떠한 typeclass 에도 속할 수 없기 때문입니다. 따라서 Parser 를 다음처럼 새로운 자료형으로 정의하도록 합니다.
```haskell
newtype Parser s a = Parser {runParser :: s -> Maybe (a, s)}
```
이제 Parser 를 Monad 의 Instance 로 만들 수 있습니다.
```haskell
instance Monad (Parser s) where
  (>>=) = parserBind
  return = parserReturn

{- 앞서 구현했던 함수와 비교해 보세요. newtype 관련 부분을 빼곤 다를 게 없습니다.
pack         a =          \input -> Just (a, input)                            -}
parserReturn a = Parser $ \input -> Just (a, input)

{- 앞서 구현했던 함수와 비교해 보세요. newtype 관련 부분을 빼곤 다를 게 없습니다.
andThen    parse next =          \input ->
  case           parse input of
    Nothing          -> Nothing
    Just (a, input') ->            next a  input'                              -}
parserBind parse next = Parser $ \input ->
  case runParser parse input of
    Nothing          -> Nothing
    Just (a, input') -> runParser (next a) input'
```
(연습) Parser 를 Monad 의 Instance 로 만들었기 때문에 Parser 를 Applicative 와 Functor 의 instance 로도 만들어야 합니다. 아래 코드에서 Functor 로 만드는 코드를 완성해 보세요.
```haskell
import Control.Monad (ap)

instance Functor (Parser s) where
  fmap f parser = Parser $ \input -> ?

instance Applicative (Parser s) where
  pure = return
  (<*>) = ap
```
stringBegins 함수와 number 함수도 바뀐 사항에 맞추어 다시 작성합니다.
```haskell
import Data.List (stripPrefix, span)
import Data.Char (isDigit)

stringBegins :: String -> Parser String String
stringBegins pattern = Parser $ \input ->
  case stripPrefix pattern input of
    Nothing   -> Nothing
    Just rest -> Just (pattern, rest)

number:: Parser String Int
number = Parser $ \input ->
  let (h, t) = span isDigit input
    in if h == "" then Nothing else Just (read h::Int, t)
```
이제 지금까지의 구현 사항을 이용하여 version 함수를 다음처럼 do notation 을 써서 더 간단하게 만들 수 있습니다.
```haskell
version3 = do
  stringBegins "version "
  major <- number
  stringBegins "."
  minor <- number
  stringBegins "."
  revision <- number
  return (major,minor,revision)
```
또는 Applicative 임을 이용하여 다음처럼 할 수도 있습니다.
```haskell
version4 = (,,) <$>
           (stringBegins "version " *> number <* stringBegins ".") <*>
           (number <* stringBegins ".") <*>
           number
```

    > runParser version4 "version 8.0.1"
    Just ((8,0,1),"")

추가로 하나 더 해볼것은 첫번째 Parser 가 실패했을 때 두번째 Parser 를 이용하도록 하는 것입니다. 즉, Parser 를 Alternative typeclass 의 Instance 로 만드는 것입니다. 먼저 Alternative typeclass 의 정의를 보겠습니다.
```haskell
class Applicative f => Alternative f where
  empty :: f a
  (<|>) :: f a -> f a -> f a
```
이제 다음처럼 empty 함수와 (<|>) 함수를 구현합니다.
```haskell
import Control.Applicative

instance Alternative (Parser s) where
  empty = Parser $ \_ -> Nothing

  f <|> g = Parser $ \input ->
    case runParser f input of
      Nothing -> runParser g input
      result  -> result
```
이렇게 했을 때의 동작은 다음과 같습니다.

    > let p = stringBegins "foo" <|> stringBegins "bar"
    > runParser p "foowhee"
    Just ("foo","whee")
    > runParser p "quuxly"
    Nothing
    > runParser p "barely"
    Just ("bar","ely")

여기까지 다룬 내용이 바로 Parsec 라이브러리가 어떻게 구현되어 있는지에 대한 간략한 소개입니다. 이제 실제로 Parsec 을 이용하는 코드를 살펴보겠습니다.

####Parsec combinators
우선 다음의 combinator 에 대해 알아보겠습니다.

- char
- many, many1
- anyChar
- letter, digit
- noneOf, oneOf
- string

    > import Text.Parsec
    > :set -XFlexibleContexts
    > let aParser = char 'a'
    > parse aParser "STDIN.." "abc"
    Right 'a'

위 코드에서는 Text.Parsec 모듈에서 제공하는 **char** parser 를 이용하여 문자 'a' 를 파싱하는 aParser 를 하나 만들었습니다. 그리고 parse 함수를 이용하여 aParser 를 실행하였습니다. 입력 "abc" 를 주었더니 이를 파싱하여 결과로 Right 'a' 를 내놓았습니다. 참고로 parse 함수의 세 개의 인자를 받는데 각각이 뜻하는 바는 다음과 같습니다.

    parse parser filePath input

두번째 인자의 경우(위의 예시 코드에서 "STDIN.."에 해당하는 인자)는 input 의 파일경로를 뜻하며 파싱 실패시 에러메시지에서 사용합니다. 파싱이 실패하는 다음 경우에 이를 확인할 수 있습니다.

    > parse aParser "STDIN.." "bc"
    Left "STDIN.." (line 1, column 1):
    unexpected "b"
    expecting "a"

**char** parser 는 문자 하나를 파싱하는데 "aaa" 같은 같은 문자가 여러개 있는 것을 파싱하려면 **many** parser 와 조합하여 사용하면 됩니다.

    > parse (many aParser) "STDIN.." "aaabc"
    Right "aaa"
    > parse (many aParser) "STDIN.." "xyz"
    Right ""

여기서 한 가지 알 수 있는 점은 **many** parser 는 절대로 실패하는 경우가 없다는 점입니다. 일치하는 경우가 전혀 없을 때는 실패하는 대신 빈 결과를 돌려줍니다. 적어도 하나 일치하는 경우가 있어야만 파싱이 성공하는 것으로 처리하고 싶을 때는 **many1** parser 를 씁니다.

    > parse (many1 aParser) "STDIN.." "xyz"
    Left "STDIN.." (line 1, column 1):
    unexpected "x"
    expecting "a"

문자로 된 것은 모두 파싱하려면 **anyChar** parser 를 이용합니다.

    > parse (many anyChar) "STDIN.." "abc"
    Right "abc"
    > parse anyChar "STDIN.." "xyz"
    Right "x"

모든 문자 말고 영문자만 파싱하려면 **letter**, 숫자만 파싱하려면 **digit** parser 를 이용합니다.

    > parse (many letter) "STDIN.." "ab23c"
    Right "ab"
    > parse (many digit) "STDIN.." "12ab3"
    Right "12"

몇몇만 제외하고 파싱할 때는 **noneOf** parser 를 이용합니다.

    > parse (many $ noneOf "., ") "STDIN.." "abc xyz"
    Right "abc"

이와 반대되는 것으로 **oneOf** parser 가 있습니다.

    > parse (many $ oneOf "abc") "STDIN.." "cba xyz"
    Right "cba"

특정 문자열을 파싱하려면 **string** parser 를 이용합니다.

    > parse (string "abc") "STDIN.." "abcdef"
    Right "abc"

이제 이 combinator 들을 이용하여 작은 프로그램을 만들어보겠습니다.

####CSV parser 만들기
parseCSV 함수는 CSV 문자열을 받아서 이를 List of List of String 으로 파싱하도록 하겠습니다. 아래 그림처럼.
<img src="parseCSV.png">
다음은 parseCSV 함수의 type입니다. 파싱이 실패할 경우에는 parse error 를 보여주는데 type 은 Text.Parsec.Error 모듈에 미리 정의되어 있는 ParseError 자료형입니다.
```haskell
parseCSV :: String -> Either ParseError [[String]]
```

