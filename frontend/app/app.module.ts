// modules (Angular 2, .rc6)
import {NgModule}      from '@angular/core';
import {BrowserModule} from '@angular/platform-browser';
import { HttpModule } from '@angular/http';

// components
import {AppComponent} from './app.component';
import {WebCamComponent} from './webcam/webcam.component';

@NgModule({
    imports: [BrowserModule, HttpModule],
    declarations: [AppComponent, WebCamComponent],
    providers: [],
    bootstrap: [AppComponent]
})
export class AppModule {
}
