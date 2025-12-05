// 粘贴图片功能：监听页面粘贴事件，自动填充文件输入框
document.addEventListener('DOMContentLoaded', function() {
    const photoInput = document.getElementById('photoInput');
    const photoForm = document.getElementById('photoForm');

    // 监听整个页面的粘贴事件
    document.addEventListener('paste', function(e) {
        const items = e.clipboardData.items;
        
        // 遍历剪贴板中的项目，查找图片
        for (let i = 0; i < items.length; i++) {
            const item = items[i];
            
            // 检查是否是图片类型
            if (item.type.indexOf('image') !== -1) {
                e.preventDefault(); // 阻止默认粘贴行为
                
                const file = item.getAsFile();
                
                // 验证文件类型
                const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png'];
                if (!allowedTypes.includes(file.type)) {
                    alert('仅支持 JPG / PNG 格式的图片');
                    return;
                }
                
                // 创建 DataTransfer 对象，用于设置文件输入框的值
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                photoInput.files = dataTransfer.files;
                
                // 显示提示信息
                showPasteSuccess();
                
                // 可选：自动提交表单（如果不想自动提交，可以注释掉下面这行）
                // photoForm.submit();
                
                break; // 只处理第一个图片
            }
        }
    });
    
    // 显示粘贴成功的提示
    function showPasteSuccess() {
        // 移除之前的提示（如果存在）
        const existingAlert = document.querySelector('.paste-success-alert');
        if (existingAlert) {
            existingAlert.remove();
        }
        
        // 创建新的提示
        const alert = document.createElement('div');
        alert.className = 'alert alert-info paste-success-alert mt-2';
        alert.innerHTML = '✅ 已从剪贴板粘贴图片，请选择背景颜色后点击"一键生成证件照"按钮';
        photoInput.parentElement.appendChild(alert);
        
        // 3秒后自动消失
        setTimeout(() => {
            alert.remove();
        }, 3000);
    }
});

document.addEventListener('DOMContentLoaded', function() {
    const previewBtn = document.getElementById('previewBtn');
    const photoForm = document.getElementById('photoForm');
    const photoInput = document.getElementById('photoInput');
    const previewArea = document.getElementById('previewArea');
    const previewImage = document.getElementById('previewImage');

    if(previewBtn){
        previewBtn.addEventListener('click', function(e) {
            //验证是否选择了文件
            if(!photoInput.files || photoInput.files.length ===0){
                alert('请先选择或粘贴一张图片');
                return;
            }
            // 获取背景颜色
            const bgColor = document.querySelector('input[name="bg_color"]:checked').value;
            //创建FormData
            const formData = new FormData();
            formData.append('photo', photoInput.files[0]);
            formData.append('bg_color', bgColor);

            // 显示加载状态
            previewBtn.disabled = true;
            previewBtn.textContent = '处理中...';
            // 发送 AJAX 请求
            fetch('/preview', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                previewBtn.disabled = false;
                previewBtn.textContent = '预览证件照';

                if (data.error) {
                    alert(data.error);
                    return;
                }

                if (data.success && data.preview_base64) {
                    // 显示预览
                    previewImage.src = `data:${data.preview_mime_type};base64,${data.preview_base64}`;
                    previewArea.style.display = 'block';

                    // 滚动到预览区域
                    previewArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            })
            .catch(error => {
                previewBtn.disabled = false;
                previewBtn.textContent = '预览证件照';
                alert('预览失败，请稍后重试');
                console.error('Error:', error);
            });
        });
    }
});