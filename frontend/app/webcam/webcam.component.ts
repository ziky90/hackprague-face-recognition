import {Component, OnInit, ElementRef} from '@angular/core';
import {Http, Response, Headers, RequestOptions} from '@angular/http';
import {DomSanitizer} from '@angular/platform-browser';
import 'rxjs/Rx' ;
import {Observable} from "rxjs/Observable";
import {IAnalysis} from "./analysis";



@Component({
    moduleId: module.id,
    selector: 'webcam-component',
    templateUrl: 'webcam.html',
    styleUrls: ['webcam.css']
})


export class WebCamComponent implements OnInit {
    public videosrc:any;
    public analysis:IAnalysis;

    constructor(private sanitizer:DomSanitizer,
                private element:ElementRef,
                private _http: Http)
                { }

    ngOnInit() {
        this.showCam();
    }


    public openWindow(){
        console.info("HELLO")
        window.open(this.getPictureURL())
    }


    private getPictureURL() {
        const video = this.element.nativeElement.querySelector('video');
        //const canvas = document.createElement('canvas');
        const canvas = this.element.nativeElement.querySelector('canvas');
        if (video) {
            canvas.width = 480;
            canvas.height = 480;
            let ctx = canvas.getContext("2d");
            ctx.drawImage(video, 0, 0);
            console.log(canvas)
            let dataURL = canvas.toDataURL("image/png");
            return dataURL
            /*canvas.msGetInputContext('2d').drawImage(video, 480, 480).subscribe(data => this.downloadFile(data)),
             error => console.log("Error downloading the file."),
             () => console.info("OK");*/
        }
    }

    public analyzeImage() {
        console.log(this.analyseImage())
    }


    public analyseImage(): Observable<IAnalysis> {
        let headers = new Headers({ 'Content-Type': 'application/json' });
        let options = new RequestOptions({ headers: headers });
        return this._http.post("localhost:5000/analyse_image","{'image_path':'/Users/jakubbares/Downloads.save.png'}",options)
            .map((response: Response) => <IAnalysis> response.json())
            .do(data => this.analysis = JSON.parse(JSON.stringify(data))[0])
            .catch(this.handleError);
    }

    private handleError(error: Response) {
            console.error(error);
            return Observable.throw(error.json().error || 'Server error')
        }


    private showCam() {
        // 1. Casting necessary because TypeScript doesn't
        // know object Type 'navigator';
        let nav = <any>navigator;

        // 2. Adjust for all browsers
        nav.getUserMedia = nav.getUserMedia || nav.mozGetUserMedia || nav.webkitGetUserMedia;

        // 3. Trigger lifecycle tick (ugly, but works - see (better) Promise example below)
        //setTimeout(() => { }, 100);

        // 4. Get stream from webcam
        nav.getUserMedia(
            {video: true},
            (stream) => {
                let webcamUrl = URL.createObjectURL(stream);

                // 4a. Tell Angular the stream comes from a trusted source
                this.videosrc = this.sanitizer.bypassSecurityTrustUrl(webcamUrl);

                // 4b. Start video element to stream automatically from webcam.
                this.element.nativeElement.querySelector('video').autoplay = true;
            },
            (err) => console.log(err));


        // OR: other method, see http://stackoverflow.com/questions/32645724/angular2-cant-set-video-src-from-navigator-getusermedia for credits
        var promise = new Promise<string>((resolve, reject) => {
            nav.getUserMedia({video: true}, (stream) => {
                resolve(stream);
            }, (err) => reject(err));
        }).then((stream) => {
            let webcamUrl = URL.createObjectURL(stream);
            this.videosrc = this.sanitizer.bypassSecurityTrustResourceUrl(webcamUrl);
            // for example: type logic here to send stream to your  server and (re)distribute to
            // other connected clients.
        }).catch((error) => {
            console.log(error);
        });
    }
}
