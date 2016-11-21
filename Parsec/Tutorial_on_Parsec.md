###Parsec 배우기
Parsec 라이브러리를 쓰면 파싱(parsing) 작업을 매우 효율적으로 처리할 수 있습니다. 먼저 파싱이란 무엇인지에 대해 생각해 봅시다.

파싱을 아주 단순하게 표현해 보면, 임의의 문자열을 입력으로 받아서 해당 문자열에서 특정 pattern 을 찾아내는 것이라고 할 수 있습니다. 조금 더 설명을 덧붙여 보면 입력 문자열에서 특정 pattern 을 찾을 수 있을 만큼 찾고나서 찾은 pattern 과 분석하지 못하고 남은 문자열을 함께 돌려주는 것이라고 할 수 있습니다. 이 설명을 type 으로 옮겨보면 다음과 같습니다.
```haskell
type Parser s a = s -> (a, s)
```
여기서 s 는 입력을, a 는 찿고자 하는 pattern 을 뜻합니다. 그런데 하나 더 고려해야 할 게 있습니다. 뭘까요? 파싱이 항상 성공하는게 아니라는 것이지요. 그래서 다음처럼 type 을 바꿉니다.
```haskell
type Parser s a = s -> Maybe (a, s)
```
이제 아주 간단한 Parser 하나를 만들어 보겠습니다. 입력 문자열이 특정 pattern 으로 시작하는지를 검사하는 Parser 입니다.
```
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

이렇게 만든 stringBegin, number 두 개의 파서를 가지고 다음처럼 동작하는 version 이라는 아주 간단한 파서를 만들어봅니다.

    > version "version 8.0.1"
    (8,0,1)

단순하게는 다음처럼 만들 수 있습니다.
```haskell
versionDumb i0 =
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
이제 좀 더 똑똑한 방법으로 다시 작성하려보니 위 코드는 다음과 같은 패턴의 반복임을 알 수 있습니다.

Parser 에 입력을 준다 → Parser 는 입력을 분석해서 찾으려는 패턴을 찾는다 → 찾으려는 패턴을 찾아서 분석이 성공하면 입력에서 남은 부분을 다음 Parser 로 넘긴다


