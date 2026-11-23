import fs from 'node:fs/promises';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
// Run from repository root with @oai/artifact-tool installed.
const dir=process.cwd();
const data=JSON.parse(await fs.readFile(dir+'/02-Multimodal/phase2-sglang/slides/deck_content.json','utf8'));
const deck=Presentation.create({slideSize:{width:1280,height:720}});
function text(slide,content,x,y,w,h,size,color,bold=false){
 const box=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 box.text=content;box.text.style={fontFamily:'PingFang SC',fontSize:size,color,bold};box.text.insets=0;return box;
}
for(let i=0;i<data.length;i++){
 const [title,sub,rows,notes]=data[i];const s=deck.slides.add();s.background.fill='#0F1C2D';
 if(i===0){
  text(s,'MULTIMODAL INFERENCE SYSTEMS',76,74,1100,48,24,'#7AE0CC',true);
  text(s,title,76,172,1140,190,58,'#F5F7FB',true);
  text(s,sub,80,390,1100,64,34,'#7AE0CC');
  text(s,'图形图像及多模态 · 课程项目',80,550,1100,50,25,'#C2CCDA');
  text(s,'工程实现稿  /  测试与 GPU 实验未运行',80,610,1100,50,24,'#F3C47D');
 }else{
  text(s,sub,76,48,1080,40,23,'#7AE0CC',true);
  text(s,title,76,112,1130,100,44,'#F5F7FB',true);
  rows.forEach(([label,body],j)=>{
    text(s,label,80,256+j*116,240,58,28,'#7AE0CC',true);
    text(s,body,330,256+j*116,850,86,29,'#E7ECF3');
  });
  text(s,String(i+1).padStart(2,'0')+' / 14',1090,650,120,35,20,'#8E9CAF');
 }
 s.speakerNotes.text=notes;
}
await (await PresentationFile.exportPptx(deck)).save(dir+'/02-Multimodal/phase2-sglang/slides/output/defense.pptx');
await fs.mkdir(dir+'/.build/previews',{recursive:true});
for(let i=0;i<deck.slides.items.length;i++){
 const blob=await deck.export({slide:deck.slides.items[i],format:'png',scale:0.6});
 await fs.writeFile(dir+'/.build/previews/slide-'+(i+1)+'.png',new Uint8Array(await blob.arrayBuffer()));
}
console.log('Created 14-slide PPTX and design previews. Software tests were not run.');
