import QtQuick 2.0
import QtQuick.Layouts 1.11
import QtQuick.Controls 2.1
import QtQuick.Window 2.1
import QtQuick.Controls.Material 2.1

ApplicationWindow {
    visible:true
    visibility: "Maximized"

    header: TabBar {
    id: bar
    width: parent.width
    TabButton {
        text: qsTr("Home")
    }
    TabButton {
        text: qsTr("Discover")
    }
    TabButton {
        text: qsTr("Activity")
    }
}

StackLayout {
    width: parent.width
    currentIndex: bar.currentIndex
    Item {
        id: homeTab
    }
    Item {
        id: discoverTab
    }
    Item {
        id: activityTab
    }
}
}

