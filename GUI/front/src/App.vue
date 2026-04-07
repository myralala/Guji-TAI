<template>
	<div id="app">
		<el-container>
			<el-header>
				<div class="leftroot">
					<div class="logo"></div>
					<div class="logotitle">
						<strong>古籍智能解读平台</strong>
					</div>
				</div>

			</el-header>

			<el-divider></el-divider>

			<el-main>
				<router-view />
			</el-main>
		</el-container>
	</div>
</template>

<script>
export default {
	data() {
		return {
			language: "zh",
		};
	},

    mounted() {
        this.$i18n.locale = this.language;
        
		document.addEventListener("DOMContentLoaded", function () {
			const linkItems = document.querySelectorAll(".headerdiv .link-item");

			linkItems.forEach((item) => {
				item.addEventListener("click", function () {
					// 移除所有元素的选中状态
					linkItems.forEach((link) => link.classList.remove("selected"));
					// 为当前点击的元素添加选中状态
					this.classList.add("selected");
				});
			});
		});
	},

	methods: {
		changeLanguage(lang) {
			this.$i18n.locale = lang;
		},
	},
};
</script>

<style>

:root {
	--guji-paper: #fffdf8;
	--guji-paper-soft: #f8f3e8;
	--guji-paper-deep: #f1e6cf;
	--guji-border: #d6c3a3;
	--guji-border-strong: #b48d5c;
	--guji-green: #2f6a62;
	--guji-brown: #7d5a3a;
	--guji-ink: #3f3429;
}

html,
body,
#app {
	background:
		radial-gradient(circle at top left, rgba(214, 195, 163, 0.16), transparent 26%),
		linear-gradient(180deg, #fffdfa 0%, #fbf8f1 100%);
	color: var(--guji-ink);
	font-family: "STSong", "Songti SC", "Noto Serif SC", serif;
}

/* 整个滚动条 */
::-webkit-scrollbar {
	width: 6px; /* 滚动条的宽度 */
	height: 6px; /* 滚动条的高度 */
}

/* 滚动条的轨道 */
::-webkit-scrollbar-track {
	background: #dcdfe6; /* 轨道的背景色 */
	border-radius: 10px; /* 轨道的圆角 */
}

/* 滚动条上的滑块 */
::-webkit-scrollbar-thumb {
	background: #CACAD0; /* 滑块的背景色 */
	border-radius: 10px; /* 滑块的圆角 */
}

/* 滑块在悬停时的样式 */
::-webkit-scrollbar-thumb:hover {
	background: #CACAD0; /* 悬停时滑块的背景色 */
}



</style>

<style scoped>

.el-main{
    padding: 0 14px 14px;
}
.el-divider--horizontal {
    display: block;
    height: 1px;
    width: 100%;
    margin: 0 18px 18px;
	background-color: rgba(180, 141, 92, 0.22);
}


.el-header {
	line-height: 84px;
	height: 84px;
	display: flex;
	justify-content: space-between;
	text-align: center;
	padding: 12px 22px 0;
	background: rgba(255, 253, 248, 0.92);
	border-bottom: 1px solid rgba(180, 141, 92, 0.18);
	box-shadow: 0 10px 30px rgba(92, 70, 42, 0.04);
}

.leftroot {
	display: flex;
	align-items: center;
	gap: 14px;
}
.logo {
    background-image: url("../static/Guji-TAI.png");
    width: 320px;
	height: 80px;
    background-size: contain; /* 等比缩放，图片完全显示 */
    background-repeat: no-repeat; /* 防止图片重复 */
    background-position: left center; /* 图片居中显示 */
	flex-shrink: 0;
	filter: drop-shadow(0 6px 12px rgba(64, 91, 77, 0.08));
}
.logotitle {
	display: none;
}


.link-item {
	height: 100%;
	padding: 5px;
	margin-left: 50px;
	border-radius: 15px;
	font-size: 1.05rem;
	color: black;
	transition: background-color 0.3s ease, color 0.3s ease;
	cursor: pointer;
	transition: background-color 0.3s, color 0.3s;
}
.link-item:hover {
	background-color: #4096ff; /* 选中时的背景色 */
	color: white; /* 选中时的字体颜色 */
}


</style>
