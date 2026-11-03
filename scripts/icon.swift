import AppKit
import Foundation
let manager = FileManager.default
try manager.createDirectory(atPath: "dist/Carter.iconset", withIntermediateDirectories: true)
for size in [16,32,64,128,256,512,1024] {
    let image = NSImage(size: NSSize(width: size, height: size))
    image.lockFocus()
    let scale = CGFloat(size) / 1024
    NSColor(calibratedRed: 0.12, green: 0.13, blue: 0.12, alpha: 1).setFill()
    NSBezierPath(roundedRect: NSRect(x: 0,y: 0,width: size,height: size), xRadius: 220*scale, yRadius: 220*scale).fill()
    let style = NSMutableParagraphStyle(); style.alignment = .center
    let text = "c" as NSString
    text.draw(in: NSRect(x: 0,y: 145*scale,width: CGFloat(size),height: 850*scale), withAttributes: [
        .font: NSFont.systemFont(ofSize: 760*scale, weight: .medium),
        .foregroundColor: NSColor(calibratedRed: 0.87,green: 0.89,blue: 0.80,alpha: 1), .paragraphStyle: style])
    image.unlockFocus()
    let bitmap = NSBitmapImageRep(data: image.tiffRepresentation!)!
    let data = bitmap.representation(using: .png, properties: [:])!
    if size <= 512 { try data.write(to: URL(fileURLWithPath: "dist/Carter.iconset/icon_\(size)x\(size).png")) }
    if size >= 32 { let half = size/2; try data.write(to: URL(fileURLWithPath: "dist/Carter.iconset/icon_\(half)x\(half)@2x.png")) }
}
let task = Process(); task.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
task.arguments = ["-c","icns","dist/Carter.iconset","-o","dist/Carter.icns"]
try task.run(); task.waitUntilExit(); if task.terminationStatus != 0 { exit(1) }
