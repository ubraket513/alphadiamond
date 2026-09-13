#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QProcess>
#include <QRegularExpression>
#include <QTemporaryDir>
#include <QTimer>
#include <QXmlStreamReader>

#include <cstdio>

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    if (argc != 4)
        return 1;
    QTemporaryDir temporary;
    QFile resource_file(QString::fromLocal8Bit(argv[3]));
    if (!temporary.isValid() || !resource_file.open(QIODevice::ReadOnly))
        return 1;
    QString resources = QString::fromUtf8(resource_file.readAll());
    QXmlStreamReader xml(resources);
    QString original_main;
    while (!xml.atEnd()) {
        xml.readNext();
        if (xml.isStartElement() && xml.name() == QStringLiteral("file") &&
            xml.attributes().value(QStringLiteral("alias")) == QStringLiteral("Main.qml"))
            original_main = xml.readElementText();
    }
    QFile source(original_main);
    if (xml.hasError() || original_main.isEmpty() || !source.open(QIODevice::ReadOnly))
        return 1;
    const QByteArray original = source.readAll();
    const QString temporary_main = QDir::fromNativeSeparators(temporary.filePath("Main.qml"));
    const QString temporary_resources = temporary.filePath("preview.qrc");
    const auto write = [](const QString& path, const QByteArray& bytes) {
        QFile file(path);
        return file.open(QIODevice::WriteOnly | QIODevice::Truncate) &&
               file.write(bytes) == bytes.size();
    };
    resources.replace(original_main.toHtmlEscaped(), temporary_main.toHtmlEscaped());
    if (!write(temporary_main, original) || !write(temporary_resources, resources.toUtf8()))
        return 1;

    QProcess preview;
    preview.setProcessChannelMode(QProcess::MergedChannels);
    QByteArray output;
    bool edited = false;
    QObject::connect(&preview, &QProcess::readyReadStandardOutput, &app, [&] {
        const QByteArray chunk = preview.readAllStandardOutput();
        output += chunk;
        std::fwrite(chunk.constData(), 1, static_cast<size_t>(chunk.size()), stdout);
        std::fflush(stdout);
        if (!edited && output.contains("preview check: ready for a source edit")) {
            edited = true;
            QTimer::singleShot(500, &app, [&] {
                QString changed = QString::fromUtf8(original);
                changed.replace(QRegularExpression(QStringLiteral("^    title:.*$"),
                                                   QRegularExpression::MultilineOption),
                                QStringLiteral("    title: \"Diamond hot reload verified\""));
                if (changed.toUtf8() == original || !write(temporary_main, changed.toUtf8()))
                    app.exit(1);
            });
        }
    });
    QObject::connect(&preview, &QProcess::finished, &app,
                     [&](int code, QProcess::ExitStatus status) {
                         const bool passed =
                             status == QProcess::NormalExit && code == 0 && edited &&
                             output.contains("source edit applied in place, UI state preserved=1");
                         app.exit(passed ? 0 : 1);
                     });
    QObject::connect(&preview, &QProcess::errorOccurred, &app, [&] { app.exit(1); });
    preview.start(QString::fromLocal8Bit(argv[1]),
                  {QStringLiteral("--verbose"), QStringLiteral("--resource"), temporary_resources,
                   QString::fromLocal8Bit(argv[2]), QStringLiteral("--preview-check")});
    QTimer::singleShot(40000, &app, [&] { app.exit(1); });
    const int result = app.exec();
    if (preview.state() != QProcess::NotRunning) {
        // Ask qmlpreview to quit so it also tears down its child application.
        preview.write("quit\n");
        preview.waitForFinished(3000);
    }
    return result;
}
