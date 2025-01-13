import QtQuick
import QtQuick.Controls

ApplicationWindow {
    visible: true
    width: 400
    height: 600
    x: screen.desktopAvailableWidth - width - 12
    y: screen.desktopAvailableHeight - height - 48
    title: "Clock"
    // flags: Qt.FramelessWindowHint | Qt.Window
    // property string currTime: "00:00:00"
    property QtObject backend
    
    Connections {
        target: backend

        function onUpdatedFrame(frame) {
            // currTime = msg;
        }
    }
    Image {
        source: "image://live/image"

    }
}