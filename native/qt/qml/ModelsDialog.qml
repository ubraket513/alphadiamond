import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import Style

AppDialog {
    id: root

    required property var controller
    readonly property var catalog: controller.modelCatalog

    objectName: "modelsDialog"
    title: "Models"
    message: "Browse native models from GitHub and Hugging Face. "
             + "A selection becomes active when the next game starts."
    acceptText: "Close"
    showReject: false
    implicitWidth: 780

    onOpened: catalog.refresh()

    ColumnLayout {
        Layout.fillWidth: true
        Layout.preferredWidth: 720
        spacing: Theme.spacing

        Rectangle {
            Layout.fillWidth: true
            implicitHeight: 78
            radius: Theme.radiusSmall
            color: Theme.surfaceAlt
            border.width: 1
            border.color: Theme.border

            RowLayout {
                anchors.fill: parent
                anchors.margins: Theme.spacingLarge
                spacing: Theme.spacingLarge

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 3

                    Text {
                        text: "ACTIVE NOW"
                        color: Theme.textFaint
                        font.family: Theme.fontFamily
                        font.pixelSize: Theme.fontTiny
                        font.weight: Theme.weightMedium
                    }
                    Text {
                        text: root.catalog.activeModelLabel
                        color: Theme.text
                        font.family: Theme.fontFamily
                        font.pixelSize: Theme.fontBody
                        font.weight: Theme.weightBold
                    }
                }

                Rectangle {
                    implicitWidth: 1
                    Layout.fillHeight: true
                    color: Theme.border
                }

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 3

                    Text {
                        text: "SELECTED FOR NEXT GAME"
                        color: Theme.textFaint
                        font.family: Theme.fontFamily
                        font.pixelSize: Theme.fontTiny
                        font.weight: Theme.weightMedium
                    }
                    Text {
                        text: root.catalog.selectedModelLabel
                        color: Theme.selection
                        font.family: Theme.fontFamily
                        font.pixelSize: Theme.fontBody
                        font.weight: Theme.weightBold
                    }
                }

                ActionButton {
                    text: root.catalog.busy ? "Refreshing…" : "Refresh"
                    enabled: !root.catalog.busy
                    onClicked: root.catalog.refresh()
                }
            }
        }

        Text {
            Layout.fillWidth: true
            text: root.catalog.status
            color: Theme.textMuted
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontSmall
            wrapMode: Text.WordWrap
        }

        ListView {
            id: modelList
            Layout.fillWidth: true
            Layout.preferredHeight: 390
            clip: true
            spacing: Theme.spacing
            model: root.catalog.models
            ScrollBar.vertical: PanelScrollBar {}

            delegate: Rectangle {
                required property var modelData

                width: modelList.width - (modelList.ScrollBar.vertical.visible ? 10 : 0)
                height: 104
                radius: Theme.radiusSmall
                color: Theme.surface
                border.width: 1
                border.color: modelData.selected ? Theme.selection : Theme.border

                RowLayout {
                    anchors.fill: parent
                    anchors.margins: Theme.spacingLarge
                    spacing: Theme.spacingLarge

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 4

                        RowLayout {
                            spacing: Theme.spacing

                            Text {
                                text: modelData.name
                                color: Theme.text
                                font.family: Theme.fontFamily
                                font.pixelSize: Theme.fontBody
                                font.weight: Theme.weightBold
                            }

                            Image {
                                visible: modelData.github
                                source: "qrc:/assets/models/github.svg"
                                sourceSize.width: 16
                                sourceSize.height: 16
                                Layout.preferredWidth: 16
                                Layout.preferredHeight: 16
                            }

                            Image {
                                visible: modelData.huggingFace
                                source: "qrc:/assets/models/huggingface.svg"
                                sourceSize.width: 16
                                sourceSize.height: 16
                                Layout.preferredWidth: 16
                                Layout.preferredHeight: 16
                            }

                            Text {
                                visible: modelData.active
                                text: "ACTIVE"
                                color: Theme.success
                                font.family: Theme.fontFamily
                                font.pixelSize: Theme.fontTiny
                                font.weight: Theme.weightBold
                            }

                            Text {
                                visible: modelData.selected && !modelData.active
                                text: "NEXT GAME"
                                color: Theme.selection
                                font.family: Theme.fontFamily
                                font.pixelSize: Theme.fontTiny
                                font.weight: Theme.weightBold
                            }
                        }

                        Text {
                            text: "Training steps: " + modelData.trainingStep
                                  + "  ·  Latest ELO: "
                                  + (modelData.latestElo === "" ? "—" : modelData.latestElo)
                                  + "  ·  Training simulations: "
                                  + modelData.trainingSimulations
                            color: Theme.textMuted
                            font.family: Theme.fontFamily
                            font.pixelSize: Theme.fontSmall
                            elide: Text.ElideRight
                            Layout.fillWidth: true
                        }

                    }

                    ActionButton {
                        visible: modelData.githubUrl !== "" || modelData.huggingFaceUrl !== ""
                        text: "View"
                        onClicked: Qt.openUrlExternally(modelData.huggingFaceUrl !== ""
                            ? modelData.huggingFaceUrl : modelData.githubUrl)
                    }

                    ActionButton {
                        text: modelData.selected ? "Selected"
                              : modelData.installed ? "Use next game" : "Download"
                        primary: modelData.installed && !modelData.selected
                        enabled: !root.catalog.busy && !modelData.selected
                                 && (modelData.installed || modelData.github || modelData.huggingFace)
                        onClicked: {
                            if (modelData.installed)
                                root.catalog.selectModel(modelData.id)
                            else
                                root.catalog.downloadModel(modelData.id)
                        }
                    }
                }
            }
        }

        Text {
            Layout.fillWidth: true
            visible: root.catalog.models.length === 0 && !root.catalog.busy
            text: "No model artifacts were found."
            color: Theme.textMuted
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontBody
            horizontalAlignment: Text.AlignHCenter
        }

        Text {
            Layout.fillWidth: true
            text: "Local storage: " + root.catalog.localRoot
            color: Theme.textFaint
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontTiny
            elide: Text.ElideMiddle
        }
    }
}
