//
//  ContentView.swift
//  Face Anonymizer
//
//  Created by Jose Bouza on 3/31/20.
//  Copyright © 2020 Jose Bouza. All rights reserved.
//

import SwiftUI

struct CaptureImageView {

        /// MARK: - Properties
        @Binding var isShown: Bool
        @Binding var image: Image?
        @Binding var size : CGSize?

        func makeCoordinator() -> Coordinator {
            return Coordinator(isShown: $isShown, image: $image, size: $size)
    }
}


struct ContentView: View {
    
    @State var image: Image? = nil
    @State var showCaptureImageView: Bool = false
    @State var size: CGSize? = nil
    
    var body: some View {
        ZStack {
          VStack {
            Button(action: {
              self.showCaptureImageView.toggle()
            }) {
              Text("Choose photo to anonymize")
            }
            
            image?.resizable().frame(width: size?.width, height: size?.height)
            
            }
          if (showCaptureImageView) {
            CaptureImageView(isShown: $showCaptureImageView, image: $image, size: $size)
          }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

