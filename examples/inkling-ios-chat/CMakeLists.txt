cmake_minimum_required(VERSION 3.21)
project(InklingChat LANGUAGES C CXX OBJCXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS" OR NOT CMAKE_OSX_SYSROOT MATCHES "[Ii][Pp]hone[Oo][Ss]")
    message(FATAL_ERROR "Configure with -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos for a physical iPhone.")
endif()

set(engine_source "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(BUILD_SHARED_LIBS OFF)
set(GGML_BLAS OFF)
set(GGML_METAL ON)
set(GGML_METAL_EMBED_LIBRARY ON)
set(GGML_OPENMP OFF)
add_subdirectory("${engine_source}" engine EXCLUDE_FROM_ALL)

set(INKLING_IQ1S_BUNDLE_ID "com.atomicengine.inklingiq1" CACHE STRING "IQ1_S iOS bundle identifier")
set(INKLING_IQ2M_BUNDLE_ID "com.atomicengine.inklingchat" CACHE STRING "IQ2_M iOS bundle identifier")

function(add_inkling_app target source_dir bundle_id display_name)
    add_executable(${target} MACOSX_BUNDLE "${source_dir}/main.mm" "${source_dir}/Inkling.mm")
    target_include_directories(${target} PRIVATE "${engine_source}/src")
    target_compile_options(${target} PRIVATE -fobjc-arc)
    target_link_libraries(${target} PRIVATE llama "-framework UIKit" "-framework Foundation")
    set_target_properties(${target} PROPERTIES
        OBJCXX_STANDARD 17
        OBJCXX_STANDARD_REQUIRED YES
        MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/DeviceInfo.plist.in"
        MACOSX_BUNDLE_GUI_IDENTIFIER "${bundle_id}"
        MACOSX_BUNDLE_BUNDLE_NAME "${display_name}"
        XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++17"
        XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "${bundle_id}"
        XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY 1
        XCODE_ATTRIBUTE_CODE_SIGN_STYLE Automatic
        XCODE_GENERATE_SCHEME TRUE)
endfunction()

add_inkling_app(InklingIQ2M . "${INKLING_IQ2M_BUNDLE_ID}" "Inkling Chat")
add_inkling_app(InklingIQ1S iq1 "${INKLING_IQ1S_BUNDLE_ID}" "Inkling Q1")
