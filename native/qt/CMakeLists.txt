find_package(Qt6 REQUIRED COMPONENTS Core Gui Qml Quick QuickControls2 Multimedia)

if(DIAMOND_BUILD_QT_SOO)
    find_package(Torch REQUIRED)
    if(NOT TARGET alphadiamond::diamond_model)
        message(FATAL_ERROR "DIAMOND_BUILD_QT_SOO requires DIAMOND_BUILD_LIBTORCH_PROBE=ON")
    endif()
endif()

qt_add_executable(diamond_qt
    main.cpp
    ai_worker.cpp
    ai_worker.hpp
    native_controller.cpp
    native_controller.hpp
    native_move_player.cpp
    native_move_player.hpp
    native_chrome.cpp
    native_chrome.hpp
)

set(QML_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../src/diamond/qml")
set(ASSET_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../src/diamond/assets")

qt_add_resources(diamond_qt qml_resources
    PREFIX "/qml"
    BASE "${QML_ROOT}"
    FILES
        "${QML_ROOT}/Main.qml"
        "${QML_ROOT}/ActionButton.qml"
        "${QML_ROOT}/AppDialog.qml"
        "${QML_ROOT}/Board.qml"
        "${QML_ROOT}/GamePanel.qml"
        "${QML_ROOT}/AiPanel.qml"
        "${QML_ROOT}/Hole.qml"
        "${QML_ROOT}/HistoryPanel.qml"
        "${QML_ROOT}/Icon.qml"
        "${QML_ROOT}/NewMatchDialog.qml"
        "${QML_ROOT}/PanelScrollBar.qml"
        "${QML_ROOT}/PanelSection.qml"
        "${QML_ROOT}/Piece.qml"
        "${QML_ROOT}/ReorderButton.qml"
        "${QML_ROOT}/ResultDialog.qml"
        "${QML_ROOT}/SegmentedControl.qml"
        "${QML_ROOT}/SidePanel.qml"
        "${QML_ROOT}/SoundDialog.qml"
        "${QML_ROOT}/TitleBar.qml"
        "${QML_ROOT}/TitleMenu.qml"
        "${QML_ROOT}/WindowButton.qml"
        "${QML_ROOT}/Style/Theme.qml"
        "${QML_ROOT}/Style/qmldir"
)

qt_add_resources(diamond_qt font_resources
    PREFIX "/fonts"
    BASE "${ASSET_ROOT}/fonts"
    FILES
        "${ASSET_ROOT}/fonts/GoogleSansFlex-Regular.ttf"
        "${ASSET_ROOT}/fonts/GoogleSansFlex-Medium.ttf"
        "${ASSET_ROOT}/fonts/GoogleSansFlex-Bold.ttf"
)

target_link_libraries(diamond_qt PRIVATE
    Qt6::Core Qt6::Gui Qt6::Qml Qt6::Quick Qt6::QuickControls2 Qt6::Multimedia
    alphadiamond::soo_core alphadiamond::soo_search
)
if(DIAMOND_BUILD_QT_SOO)
    target_link_libraries(diamond_qt PRIVATE alphadiamond::diamond_model ${TORCH_LIBRARIES})
    target_include_directories(diamond_qt PRIVATE ${TORCH_INCLUDE_DIRS})
    target_compile_definitions(diamond_qt PRIVATE DIAMOND_QT_HAS_SOO _DISABLE_EXTENDED_ALIGNED_STORAGE QT_NO_KEYWORDS)
endif()
target_compile_features(diamond_qt PRIVATE cxx_std_20)
set_target_properties(diamond_qt PROPERTIES AUTOMOC ON WIN32_EXECUTABLE TRUE)

if(MSVC)
    target_compile_options(diamond_qt PRIVATE /W4)
endif()
if(WIN32)
    target_link_libraries(diamond_qt PRIVATE dwmapi shell32 user32)
endif()

if(BUILD_TESTING)
    add_executable(diamond_qt_controller_contract
        tests/controller_contract.cpp
        ai_worker.cpp
        ai_worker.hpp
        native_controller.cpp
        native_controller.hpp
        native_move_player.cpp
        native_move_player.hpp
    )
    target_link_libraries(diamond_qt_controller_contract PRIVATE
        Qt6::Core Qt6::Gui Qt6::Qml Qt6::Quick Qt6::Multimedia
        alphadiamond::soo_core alphadiamond::soo_search
    )
    target_include_directories(diamond_qt_controller_contract PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_compile_features(diamond_qt_controller_contract PRIVATE cxx_std_20)
    set_target_properties(diamond_qt_controller_contract PROPERTIES AUTOMOC ON)
    if(MSVC)
        target_compile_options(diamond_qt_controller_contract PRIVATE /W4)
    endif()
    add_test(NAME diamond_qt_controller_contract COMMAND diamond_qt_controller_contract)
    set_tests_properties(diamond_qt_controller_contract PROPERTIES
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
    )
endif()

set(MOVE_SOUND "${CMAKE_CURRENT_SOURCE_DIR}/../../src/diamond/assets/sounds/move.m4a")
foreach(sound_target diamond_qt diamond_qt_controller_contract)
    if(TARGET ${sound_target})
        add_custom_command(TARGET ${sound_target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${sound_target}>/assets/sounds"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${MOVE_SOUND}" "$<TARGET_FILE_DIR:${sound_target}>/assets/sounds/move.m4a"
            VERBATIM)
    endif()
endforeach()
