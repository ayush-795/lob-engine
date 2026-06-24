cmake_minimum_required(VERSION 3.16)
project(lob_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
add_compile_options(-Wall -Wextra)

add_library(lob STATIC src/order_book.cpp)
target_include_directories(lob PUBLIC include)

add_executable(lob_replay bench/replay.cpp)
target_link_libraries(lob_replay PRIVATE lob)

enable_testing()
add_executable(test_order_book tests/test_order_book.cpp)
target_link_libraries(test_order_book PRIVATE lob)
add_test(NAME order_book COMMAND test_order_book)
