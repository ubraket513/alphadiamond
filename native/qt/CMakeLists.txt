find_package(Qt6 REQUIRED COMPONENTS Core Gui Qml Quick QuickControls2)

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
)

set(QML_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../src/diamond/qml")

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

target_link_libraries(diamond_qt PRIVATE
    Qt6::Core Qt6::Gui Qt6::Qml Qt6::Quick Qt6::QuickControls2
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
